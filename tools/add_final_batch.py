"""Final batch to reach 500+."""
import json, random
from pathlib import Path
random.seed(777)
import sys; sys.path.insert(0, str(Path(__file__).parent.parent))
from tools.build_dataset import mk, gen_scores, SUMMARIES

FINAL = [
("A SaaS for managing salon appointments, staff schedules, and client loyalty.","average"),
("An AI analyzing email campaigns and suggesting subject line improvements.","average"),
("A platform for managing coworking space visitor access and meeting rooms.","average"),
("A SaaS for managing plumbing business dispatching and invoicing.","average"),
("An AI tool generating compliance documentation from policy templates.","average"),
("A platform for managing food truck locations and menu updates.","average"),
("A SaaS for managing HVAC service contracts and preventive maintenance.","average"),
("An AI tool predicting inventory stockouts from sales velocity data.","average"),
("A platform for managing daycare center enrollment and parent communication.","average"),
("A SaaS for managing car dealership inventory and lead management.","average"),
("An AI tool generating personalized learning paths from assessment results.","average"),
("A platform for managing self-storage facility units and billing.","average"),
("A SaaS for managing laundromat operations and machine maintenance.","average"),
("An AI tool analyzing website heatmaps and suggesting UX improvements.","average"),
("A platform for managing marina slip rentals and boat maintenance.","average"),
("A SaaS automating real estate appraisal workflows and report generation.","strong"),
("An AI platform for automated medical billing denial management.","strong"),
("A developer tool providing automated chaos engineering for microservices.","strong"),
("A SaaS managing pharmaceutical clinical supply chain logistics.","strong"),
("An AI platform for real-time video analytics in retail stores.","strong"),
("A developer tool providing automated data pipeline orchestration.","strong"),
("A SaaS managing commercial insurance policy administration end-to-end.","strong"),
("An AI platform for automated customer segmentation from behavioral data.","strong"),
("A developer tool providing automated mobile app performance profiling.","strong"),
("A SaaS managing healthcare credentialing and provider enrollment.","strong"),
("An AI platform for automated supply chain demand sensing.","strong"),
("A developer tool providing automated secrets management and rotation.","strong"),
("A SaaS managing commercial lending origination and underwriting.","strong"),
("An AI platform for automated visual inspection in food processing.","strong"),
("A developer tool providing automated service mesh configuration.","strong"),
("A marketplace for local personal chefs offering in-home dining.","below_avg"),
("An app for tracking and reviewing local hiking trails.","below_avg"),
("A platform for booking mobile phone repair technicians.","below_avg"),
("An app for tracking personal meditation streaks and progress.","below_avg"),
("A marketplace for local artists offering painting classes.","below_avg"),
("A platform for booking local DJs for private events.","below_avg"),
("An app for tracking and comparing local gas prices.","below_avg"),
("A marketplace for local seamstresses offering alterations.","below_avg"),
("A platform for booking local photographers for family portraits.","below_avg"),
("An app for tracking personal gratitude journal entries.","below_avg"),
("An AI platform for real-time ocean current prediction for shipping routes.","exceptional"),
("A developer platform providing AI-powered automated penetration testing.","exceptional"),
("A fintech platform enabling instant cross-border payroll for remote teams.","exceptional"),
("An AI platform for automated drug interaction prediction from molecular data.","exceptional"),
("A SaaS providing AI-powered real-time translation for legal proceedings.","exceptional"),
("A platform for competitive speed-typing of random Wikipedia articles.","weak"),
("An app that generates a random color and tells you it matches your personality.","weak"),
("A subscription box of miniature traffic cones.","weak"),
("A SaaS that tracks how many times you open and close your laptop.","weak"),
("A marketplace for trading vintage airline sick bags.","weak"),
("An AI that writes poetry about your most recent Google search.","weak"),
("A platform for competitive speed-folding of paper airplanes.","weak"),
("An app that plays a sad trombone sound when your battery drops below 20%.","weak"),
]

full = Path(__file__).parent.parent / "data" / "training_data_full.json"
with open(full) as f:
    all_ex = json.load(f)
seen = {ex.get("source_id") for ex in all_ex}

for i, (desc, tier) in enumerate(FINAL):
    eid = f"f{i+1:03d}"
    if eid not in seen:
        sr = gen_scores(tier)
        all_ex.append(mk(eid, desc, sr, random.choice(SUMMARIES[tier])))
        seen.add(eid)

random.shuffle(all_ex)
scores = [json.loads(ex["messages"][2]["content"]).get("overall_score",0) for ex in all_ex]
print(f"Total: {len(all_ex)}")
print(f"Range: {min(scores):.1f} - {max(scores):.1f}")
print(f"Mean: {sum(scores)/len(scores):.1f}")
buckets = {"1-3":0,"3.1-5":0,"5.1-6.5":0,"6.6-8":0,"8.1-10":0}
for s in scores:
    if s<=3: buckets["1-3"]+=1
    elif s<=5: buckets["3.1-5"]+=1
    elif s<=6.5: buckets["5.1-6.5"]+=1
    elif s<=8: buckets["6.6-8"]+=1
    else: buckets["8.1-10"]+=1
print("\nDistribution:")
for b,c in buckets.items():
    pct=100*c/len(all_ex)
    print(f"  {b:>8s}: {c:>4d} ({pct:5.1f}%) {'█'*int(pct/2)}")
with open(full,"w") as f:
    json.dump(all_ex, f, indent=2, ensure_ascii=False)
print(f"\nSaved: {full}")
