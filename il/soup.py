"""Average the policies fine-tuned from one start on different replay shards (a model soup)."""
import glob, json, sys
import torch
fs = sorted(glob.glob(sys.argv[1] + "/**/policy.pt", recursive=True))
hs = [json.load(open(f.replace("policy.pt", "bc_history.json"))) for f in fs if __import__("os").path.exists(f.replace("policy.pt", "bc_history.json"))]
print("policies", len(fs))
for h in hs:
    print("  first", h[0], "last", h[-1])
sds = [torch.load(f) for f in fs]
avg = {k: (sum(sd[k].float() for sd in sds) / len(sds)).to(sds[0][k].dtype) if sds[0][k].is_floating_point() else sds[0][k] for k in sds[0]}
torch.save(avg, sys.argv[2])
json.dump(hs, open(sys.argv[2].replace(".pt", "_hist.json"), "w"))
