"""Human correction drafts. Saving labels is separate from activating a fit."""
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone

from lab_encoder import normalize_text
from .corpus import MODES
from .real_data import test_hashes


def feedback_path(root, active=False):
    return root / 'analysis_lab/private' / ('feedback.active.json' if active else 'feedback.json')


def load_feedback(root, active=False):
    path = feedback_path(root, active)
    payload = json.loads(path.read_text()) if path.is_file() else {'version': 1, 'cases': []}
    if not isinstance(payload, dict) or payload.get('version') != 1 or not isinstance(payload.get('cases'), list):
        raise ValueError('纠错归档格式非法，请恢复 private/history 中的版本')
    return payload['cases']


def feedback_digest(cases):
    return hashlib.sha256(json.dumps(cases, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def validate_feedback(cases, base_cases, baseline, real_cases=(), history=()):
    if len(cases) > 1000:
        raise ValueError('当前纠错归档最多 1000 条')
    heldout = {normalize_text(c['text']) for c in base_cases if c['split'] != 'train'}
    references = {normalize_text(c['text']) for c in real_cases}
    families = {c['family']: c['split'] for c in real_cases}
    ids, texts, ledger = set(), set(), set(history)
    training = {normalize_text(c['text']) for c in base_cases + baseline}
    for c in cases:
        if c.get('mode') not in MODES or c.get('split') not in {'train', 'validation', 'test'} or c.get('source') not in {'manual', 'asr'}:
            raise ValueError('纠错标签、集合或来源非法')
        if not isinstance(c.get('text'), str) or len(c['text']) > 256 or not normalize_text(c['text']):
            raise ValueError('请归档 1–256 字的实际原句')
        if not isinstance(c.get('family'), str) or not c['family'].strip() or len(c['family']) > 100:
            raise ValueError('表达家族 ID 必须为 1–100 字符')
        text = normalize_text(c['text'])
        if c.get('id') != 'feedback:' + hashlib.sha256(text.encode()).hexdigest():
            raise ValueError('纠错 ID 与标准化原句不匹配')
        if c.get('id') in ids or text in texts or text in references:
            raise ValueError('原句已在纠错/导入数据中，不能重复归档')
        if text in heldout:
            raise ValueError('固定验证/测试原句不能进入纠错训练；请另采新句')
        hashes = test_hashes({'cases': [dict(c, split='test')]})
        if c['split'] != 'test' and hashes & ledger:
            raise ValueError('曾用于测试的句子或家族不能进入训练/验证')
        if c['split'] == 'test' and text in training:
            raise ValueError('原有合成/向量样例不能冒充新留出句')
        if c['family'] in families and families[c['family']] != c['split']:
            raise ValueError('同一会话/表达家族不能跨训练、验证、测试集合')
        ids.add(c['id']); texts.add(text); families[c['family']] = c['split']
    return cases


def upsert_feedback(cases, text, mode, split, source='manual', family=''):
    cleaned = normalize_text(text)
    if not cleaned:
        raise ValueError('不能归档空句或只有唤醒词的句子')
    key = hashlib.sha256(cleaned.encode()).hexdigest()
    identifier = 'feedback:' + key
    family = family.strip() or ('utterance:' + key)
    old = next((c for c in cases if c['id'] == identifier), None)
    if old and old['split'] != split:
        raise ValueError('归档集合固定，不能移入其他集合；请按家族另采新句')
    if old and split == 'test' and (old['mode'] != mode or old['family'] != family or old['source'] != source):
        raise ValueError('回归留出标签/来源/家族已冻结，不能根据预测修改')
    if old and all(old[k] == v for k, v in {'mode': mode, 'split': split, 'source': source, 'family': family}.items()) and normalize_text(old['text']) == cleaned:
        return cases, old, False
    stamp = datetime.now(timezone.utc).isoformat()
    record = {'id': identifier, 'text': text, 'normalized_text': cleaned, 'mode': mode, 'split': split, 'source': source,
              'family': family, 'subtype': '人工纠错', 'created_at': old['created_at'] if old else stamp,
              'updated_at': stamp}
    result = [record if c['id'] == identifier else c for c in cases]
    if old is None: result.append(record)
    return result, record, old is None


def feedback_summary(cases, active):
    return {'cases': cases, 'count': len(cases), 'sha256': feedback_digest(cases),
            'active_sha256': feedback_digest(active), 'active_count': len(active),
            'pending': feedback_digest(cases) != feedback_digest(active),
            'counts': {s: {m: sum(c['split'] == s and c['mode'] == m for c in cases) for m in MODES}
                       for s in ('train', 'validation', 'test')},
            'sources': dict(Counter(c['source'] for c in cases)),
            'note': '归档不立即生效；点击重新训练才更新两个本地分类器。手动输入不等于真实 ASR。回归留出来自已查看预测的探针，不是新盲测。默认每句一个家族，相近改写/同会话请填同一个稳定家族 ID。'}
