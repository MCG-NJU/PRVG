import json
import functools

def load_jsonl(filename):
    with open(filename, "r") as f:
        return [json.loads(l.strip("\n")) for l in f.readlines()]

def compare(x, y):
    # "vid": "C0MQLh8Az7U_360.0_510.0"
    start_1 = float(x['vid'].split("_")[-2])
    end_1 = float(x['vid'].split("_")[-1])

    start_2 = float(y['vid'].split("_")[-2])
    end_2 = float(y['vid'].split("_")[-1])

    if start_1 < start_2:
        return -1
    elif start_1 == start_2 and end_1 < end_2:
        return -1
    elif start_1 == start_2 and end_1 == end_2:
        return 0
    else:
        return 1

def compare_two(x, y):
    # "vid": "C0MQLh8Az7U_360.0_510.0"
    start_1 =x[0]
    end_1 = x[1]

    start_2 = y[0]
    end_2 = y[1]

    if start_1 < start_2:
        return -1
    elif start_1 == start_2 and end_1 < end_2:
        return -1
    elif start_1 == start_2 and end_1 == end_2:
        return 0
    else:
        return 1

if __name__ == '__main__':
    data = load_jsonl("./QVHighlights/highlight_train_release.jsonl")

    results = {}
    for d in data:
        start = d["vid"].split("_")[-2]
        end = d["vid"].split("_")[-1]
        vid = d["vid"].split("_"+start+"_"+end)[0]
        if vid not in results:
            results[vid] = []
        del d["relevant_clip_ids"]
        del d["saliency_scores"]
        results[vid].append(d)

    for k, v in results.items():
        results[k] = sorted(v, key=functools.cmp_to_key(compare))
    print(len(results.keys()))

    remove_keys = []
    for k, v in results.items():
        tmp = float(v[0]["vid"].split("_")[-2])
        for d in v:
            start = float(d["vid"].split("_")[-2])
            end = float(d["vid"].split("_")[-1])
            if start != tmp:
                print(k, v)
                print()
                remove_keys.append(k)
                break
            else:
                tmp = end
    for k in remove_keys:
        del results[k]
    print(len(results.keys()))

    for k, v in results.items():
        print(k, v)
        break





