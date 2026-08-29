"""
Adapts 25p result JSON (outcome field) → GPTFuzz-style schema expected by claude_judge_results.py
Usage: python3 adapt_25p_schema.py <input.json> <output.json>
"""
import json, sys

inp, out = sys.argv[1], sys.argv[2]
data = json.load(open(inp))
results = data.get('results', [])

adapted = []
for r in results:
    outcome = r.get('outcome', '')
    oi = r.get('outcome_iteration')
    os_ = r.get('outcome_stream')

    # Find winning stream's target response
    iters = r.get('iterations', [])
    jb_response = None
    if oi and os_:
        iter_data = iters[oi-1] if oi <= len(iters) else {}
        streams = iter_data.get('streams', [])
        stream_data = streams[os_-1] if os_ <= len(streams) else {}
        jb_response = stream_data.get('target_response') or stream_data.get('misdirected_response')

    adapted.append({
        'index': r['index'],
        'question': r.get('goal', ''),
        'target_str': r.get('target_str', ''),
        'attack_jailbreak': outcome == 'jailbroken' or outcome == 'attack_jailbreak',
        'misdirected_jailbreak': outcome == 'misdirected_jailbreak',
        'defended': outcome == 'defended',
        'outcome': outcome,
        'jailbreak_iteration': oi,
        'jailbreak_stream': os_,
        'iterations': [
            {
                'iteration': i+1,
                'jailbroken': (i+1 == oi) and outcome in ('jailbroken','attack_jailbreak'),
                'misdirected': (i+1 == oi) and outcome == 'misdirected_jailbreak',
                'target_responses': [
                    s.get('target_response') or s.get('misdirected_response','')
                    for s in itr.get('streams',[])
                ],
            }
            for i, itr in enumerate(iters)
        ],
    })

out_data = {'results': adapted, 'total': len(adapted)}
json.dump(out_data, open(out,'w'), indent=2)
print(f'Adapted {len(adapted)} results → {out}')
