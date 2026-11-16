"""User-labeled ASR corpus, kept private and separate from synthetic charts."""
import hashlib
import json
from collections import Counter

from lab_encoder import normalize_text
from .corpus import MODES


def validate_real_dataset(payload, base_cases, baseline, previous_tests=()):
    if not isinstance(payload, dict) or set(payload) != {'dataset_id', 'cases'}:
        raise ValueError('JSON 顶层只接受 dataset_id 和 cases')
    dataset_id, cases = payload['dataset_id'], payload['cases']
    if not isinstance(dataset_id, str) or not dataset_id.strip() or len(dataset_id) > 80:
        raise ValueError('dataset_id 必须为 1–80 字符')
    if not isinstance(cases, list) or not 1 <= len(cases) <= 1000:
        raise ValueError('cases 必须为 1–1000 条人工标注意图的 ASR 原句')
    ids, texts, families, output = set(), set(), {}, []
    heldout = {normalize_text(c['text']) for c in base_cases if c['split'] != 'train'}
    all_synthetic = {normalize_text(c['text']) for c in base_cases + baseline}
    previous_tests = set(previous_tests)
    for c in cases:
        keys = {'id', 'text', 'mode', 'split', 'family', 'source'}
        if not isinstance(c, dict) or set(c) != keys or any(not isinstance(v, str) for v in c.values()):
            raise ValueError('每条记录须含字符串 id/text/mode/split/family/source，不接受自动推断标签')
        if c['mode'] not in MODES or c['split'] not in {'train', 'validation', 'test'} or c['source'] != 'asr':
            raise ValueError('mode 为五类之一，split 为 train/validation/test，source 必须为 asr')
        if not c['id'].strip() or len(c['id']) > 80 or not c['family'].strip() or len(c['family']) > 100:
            raise ValueError('id/family 不能为空，长度分别不超过 80/100')
        text = normalize_text(c['text'])
        if not text or len(c['text']) > 256:
            raise ValueError('原句须为 1–256 字符，不能只有唤醒词')
        if c['id'] in ids or text in texts:
            raise ValueError('真实数据内有重复 ID 或标准化原句；同一句不可跨集合')
        text_hash = hashlib.sha256(text.encode()).hexdigest()
        family_hash = 'family:' + hashlib.sha256(c['family'].encode()).hexdigest()
        if c['split'] != 'test' and (text in heldout or text_hash in previous_tests or family_hash in previous_tests):
            raise ValueError('曾用于测试的句子不能进入训练/验证集；请保留旧留出集并另采新句')
        if c['split'] == 'test' and text in all_synthetic:
            raise ValueError('真实留出集与已查看的合成样例/原有设定词完全重复')
        if c['family'] in families and families[c['family']] != c['split']:
            raise ValueError('相同会话/表达家族不得跨训练、验证、测试集合')
        ids.add(c['id']); texts.add(text); families[c['family']] = c['split']
        output.append(dict(c, id='real:' + c['id']))
    counts = Counter((c['split'], c['mode']) for c in output)
    for split in ('train', 'validation'):
        if any(c['split'] == split for c in output) and any(counts[split, m] < 5 for m in MODES):
            raise ValueError('若提供真实训练/验证集，该集合五类各至少 5 条；建议每类 50 条以上')
    if any(c['split'] == 'train' for c in output) and not any(c['split'] == 'validation' for c in output):
        raise ValueError('补充真实训练句时也须提供独立真实验证集，不能用训练句调阈值')
    return {'dataset_id': dataset_id.strip(), 'cases': output}


def digest_dataset(payload):
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def test_hashes(payload):
    output = set()
    for c in payload['cases']:
        if c['split'] == 'test':
            output.add(hashlib.sha256(normalize_text(c['text']).encode()).hexdigest())
            output.add('family:' + hashlib.sha256(c['family'].encode()).hexdigest())
    return output


def public_template():
    return {'dataset_id': '填写采集批次名称', 'cases': []}
