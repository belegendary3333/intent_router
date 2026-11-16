"""Frozen BGE + multinomial linear head. Test vectors never enter fit/tuning."""
import hashlib
import json

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, log_loss

from .corpus import MODES


def softmax(logits, temperature=1.0):
    values = np.asarray(logits, dtype=np.float64) / temperature
    values -= values.max(axis=1, keepdims=True)
    probabilities = np.exp(values)
    return probabilities / probabilities.sum(axis=1, keepdims=True)


class FrozenIntentClassifier:
    C_GRID = (.01, .1, 1, 10, 100)
    TEMPERATURE_GRID = (.5, .75, 1, 1.25, 1.5, 2, 3, 5)
    SCORE_GRID = (0, .4, .5, .6, .7, .75, .8, .85, .9, .95)
    MARGIN_GRID = (0, .05, .1, .15, .2, .3, .4)
    TARGET_PRECISION = .95
    MIN_ACCEPTED = 3

    def fit(self, train_vectors, train_cases, validation_vectors, validation_cases, fingerprint):
        train_y = np.array([c['mode'] for c in train_cases])
        val_y = np.array([c['mode'] for c in validation_cases])
        if set(train_y) != set(MODES) or set(val_y) != set(MODES):
            raise ValueError('训练和验证集均须包含五类意图')
        best, self.search = None, []
        for c in self.C_GRID:
            model = LogisticRegression(C=c, max_iter=2000, random_state=42).fit(train_vectors, train_y)
            predicted = model.predict(validation_vectors)
            f1 = float(f1_score(val_y, predicted, labels=MODES, average='macro', zero_division=0))
            unsafe = int(np.sum((val_y == 'chat') & (predicted != 'chat')))
            self.search.append({'C': c, 'validation_macro_f1': f1, 'unsafe_chat_activation': unsafe})
            rank = (f1, -unsafe, -c)
            if best is None or rank > best[0]:
                best = (rank, model, c)
        _, model, self.C = best
        self.classes = model.classes_.tolist()
        self.coef, self.intercept = model.coef_.copy(), model.intercept_.copy()
        logits = self.logits(validation_vectors)
        self.temperature = min(self.TEMPERATURE_GRID, key=lambda t: (
            log_loss(val_y, softmax(logits, t), labels=self.classes), abs(t - 1)))
        probs = softmax(logits, self.temperature)
        top = np.argmax(probs, axis=1)
        scores = probs[np.arange(len(probs)), top]
        margins = scores - np.sort(probs, axis=1)[:, -2]
        predicted = np.array(self.classes)[top]
        self.thresholds = {}
        for mode in MODES:
            best = None
            support = int(np.sum(val_y == mode))
            for score in self.SCORE_GRID:
                for margin in self.MARGIN_GRID:
                    accept = (predicted == mode) & (scores >= score) & (margins >= margin)
                    count = int(accept.sum())
                    correct = int(np.sum(accept & (val_y == mode)))
                    precision = correct / count if count else None
                    if count < self.MIN_ACCEPTED or precision < self.TARGET_PRECISION:
                        continue
                    # Maximize retained correct decisions under a precision target;
                    # choose fewer errors, then less restrictive gates on ties.
                    rank = (correct, -(count - correct), -score, -margin)
                    if best is None or rank > best[0]:
                        best = (rank, {'min_probability': score, 'min_margin': margin,
                                       'validation_accepted': count, 'validation_precision': precision,
                                       'validation_recall': correct / support, 'disabled': False})
            self.thresholds[mode] = best[1] if best else {
                'min_probability': 1.01, 'min_margin': 1.01,
                'validation_accepted': 0, 'validation_precision': None,
                'validation_recall': 0, 'disabled': True}
        self.provenance = {
            'embedding_fingerprint': fingerprint,
            'train_ids': [c['id'] for c in train_cases],
            'validation_ids': [c['id'] for c in validation_cases],
            'train_sha256': hashlib.sha256(json.dumps(train_cases, ensure_ascii=False, sort_keys=True).encode()).hexdigest(),
            'validation_sha256': hashlib.sha256(json.dumps(validation_cases, ensure_ascii=False, sort_keys=True).encode()).hexdigest(),
        }
        self.validation_log_loss = {
            'raw': float(log_loss(val_y, softmax(logits), labels=self.classes)),
            'temperature_scaled': float(log_loss(val_y, probs, labels=self.classes)),
        }
        return self

    def logits(self, vectors):
        return np.asarray(vectors) @ self.coef.T + self.intercept

    def predict(self, vectors, gated=True):
        if len(vectors) == 0:
            return []
        probs = softmax(self.logits(vectors), self.temperature if gated else 1)
        output = []
        for row in probs:
            order = np.argsort(-row)
            mode = self.classes[order[0]]
            score, margin = float(row[order[0]]), float(row[order[0]] - row[order[1]])
            threshold = self.thresholds[mode]
            accepted = not gated or (score >= threshold['min_probability'] and margin >= threshold['min_margin'])
            output.append({'mode': mode if accepted else 'clarify', 'matched_mode': mode,
                           'score': round(score, 4), 'margin': round(margin, 4), 'accepted': bool(accepted),
                           'scores': {self.classes[i]: round(float(row[i]), 4) for i in order},
                           'score_type': 'probability', 'policy': 'gated' if gated else 'raw',
                           'threshold': threshold if gated else None,
                           'clarification': None if accepted else '你是想让我执行这个动作，还是想聊聊它？'})
        return output

    def artifact(self):
        return {'format': 'frozen-bge-linear-v1', 'classes': self.classes, 'dimension': self.coef.shape[1],
                'coef': self.coef.tolist(), 'intercept': self.intercept.tolist(), 'C': self.C,
                'temperature': self.temperature, 'thresholds': self.thresholds, 'provenance': self.provenance,
                'search': self.search, 'validation_log_loss': self.validation_log_loss,
                'target_precision': self.TARGET_PRECISION, 'min_validation_accepted': self.MIN_ACCEPTED,
                'note': 'BGE 权重冻结；C、温度和分类阈值复用小验证集，存在选择偏差。95% 是验证集接受精度目标，不是线上保证。原始合成测试集已多次查看，只作回归比较。拒识输出 clarify，不回退成聊天；未部署到机器人。'}
