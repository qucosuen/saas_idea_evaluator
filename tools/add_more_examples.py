"""Add more examples to reach 500 total."""
import json, random
from pathlib import Path
random.seed(99)

# Import from build_dataset
import sys; sys.path.insert(0, str(Path(__file__).parent.parent))
from tools.build_dataset import mk, gen_scores, SUMMARIES, DIMS, REASONS

MORE = [
# More AVERAGE
("A SaaS for managing coworking space memberships and room bookings.","average"),
("An AI tool analyzing customer churn patterns and suggesting retention actions.","average"),
("A platform for freelance graphic designers to find and manage client projects.","average"),
("A SaaS managing fleet maintenance schedules and repair histories.","average"),
("An AI generating meeting notes and action items from calendar events.","average"),
("A platform for small hotels to manage bookings across OTAs.","average"),
("A SaaS for managing employee time tracking and project billing.","average"),
("An AI tool optimizing warehouse layout based on order patterns.","average"),
("A platform connecting freelance writers with content marketing agencies.","average"),
("A SaaS for managing dental lab orders and case tracking.","average"),
("An AI tool predicting customer lifetime value from early behavior signals.","average"),
("A platform for managing multi-vendor marketplace operations.","average"),
("A SaaS for optometry practices managing appointments and prescriptions.","average"),
("An AI tool generating SEO-optimized blog posts from topic briefs.","average"),
("A platform for managing commercial cleaning service contracts.","average"),
("A SaaS for managing pest control service routes and scheduling.","average"),
("An AI tool analyzing competitor pricing and suggesting adjustments.","average"),
("A platform for managing shared office equipment and supply orders.","average"),
("A SaaS for managing landscaping business operations and crew scheduling.","average"),
("An AI tool generating customer onboarding sequences from product usage data.","average"),
("A platform for managing music venue bookings and artist payments.","average"),
("A SaaS for managing auto repair shop workflows and parts inventory.","average"),
("An AI tool predicting equipment failure from sensor data patterns.","average"),
("A platform for managing tutoring center schedules and student progress.","average"),
("A SaaS for managing photography studio bookings and client galleries.","average"),
("An AI tool generating financial reports from accounting data.","average"),
("A platform for managing pet grooming appointments and client records.","average"),
("A SaaS for managing gym memberships, class schedules, and trainer bookings.","average"),
("An AI tool analyzing support tickets to identify product improvement areas.","average"),
("A platform for managing catering company orders and delivery logistics.","average"),
# More STRONG
("A SaaS automating healthcare prior authorization workflows.","strong"),
("An AI platform for real-time credit risk assessment using alternative data.","strong"),
("A developer tool providing automated load testing and performance optimization.","strong"),
("A SaaS managing pharmaceutical distribution and cold chain compliance.","strong"),
("An AI platform for automated quality inspection in manufacturing.","strong"),
("A developer tool providing real-time collaboration for database schema design.","strong"),
("A SaaS automating commercial lease management and rent collection.","strong"),
("An AI platform for predictive analytics in retail demand planning.","strong"),
("A developer tool providing automated accessibility testing for web apps.","strong"),
("A SaaS managing multi-carrier shipping with rate optimization.","strong"),
("An AI platform for automated content moderation at scale.","strong"),
("A developer tool providing automated dependency vulnerability scanning.","strong"),
("A SaaS managing wholesale distribution orders and inventory.","strong"),
("An AI platform for real-time speech analytics in call centers.","strong"),
("A developer tool providing automated infrastructure cost allocation.","strong"),
("A SaaS managing commercial property maintenance and tenant experience.","strong"),
("An AI platform for automated invoice data extraction and matching.","strong"),
("A developer tool providing automated canary deployments and rollbacks.","strong"),
("A SaaS managing insurance agency operations and policy administration.","strong"),
("An AI platform for automated ESG data collection and reporting.","strong"),
("A developer tool providing automated API versioning and deprecation management.","strong"),
("A SaaS managing dental supply chain ordering and inventory.","strong"),
("An AI platform for automated legal research and case law analysis.","strong"),
("A developer tool providing automated cloud cost anomaly detection.","strong"),
("A SaaS managing veterinary hospital operations including surgery scheduling.","strong"),
("An AI platform for automated financial statement analysis and benchmarking.","strong"),
("A developer tool providing automated feature flag management and experimentation.","strong"),
("A SaaS managing commercial kitchen equipment maintenance and compliance.","strong"),
("An AI platform for automated patent landscape analysis and competitive intelligence.","strong"),
("A developer tool providing automated GraphQL schema stitching and federation.","strong"),
# More BELOW_AVG
("A platform for sharing and reviewing home workout routines.","below_avg"),
("An app for tracking personal water intake with gamification.","below_avg"),
("A marketplace for local babysitters with background checks.","below_avg"),
("A platform for booking mobile nail technicians.","below_avg"),
("An app for tracking and reviewing local food trucks.","below_avg"),
("A marketplace for custom 3D-printed phone accessories.","below_avg"),
("A platform for booking local fishing guides.","below_avg"),
("An app for tracking personal reading goals and book reviews.","below_avg"),
("A marketplace for local florists to sell arrangements online.","below_avg"),
("A platform for booking mobile pet grooming services.","below_avg"),
("An app for tracking personal carbon footprint from purchases.","below_avg"),
("A marketplace for local bakers to sell custom cakes.","below_avg"),
("A platform for booking local music teachers for lessons.","below_avg"),
("An app for tracking and comparing gym membership prices.","below_avg"),
("A marketplace for local handymen to find small repair jobs.","below_avg"),
("A platform for booking mobile car wash services.","below_avg"),
("An app for tracking personal sleep patterns with recommendations.","below_avg"),
("A marketplace for local caterers to find event clients.","below_avg"),
("A platform for booking local personal trainers.","below_avg"),
("An app for tracking household expenses shared among family members.","below_avg"),
# More EXCEPTIONAL
("An AI platform for real-time seismic monitoring and earthquake early warning.","exceptional"),
("A developer platform providing AI-powered automated theorem proving for software verification.","exceptional"),
("A fintech platform enabling real-time settlement of international trade invoices.","exceptional"),
("An AI platform for automated pathology slide analysis and cancer grading.","exceptional"),
("A SaaS providing AI-powered predictive maintenance for commercial aviation fleets.","exceptional"),
("An AI platform for real-time wildfire detection and spread prediction from satellite data.","exceptional"),
("A developer platform providing quantum-resistant encryption infrastructure.","exceptional"),
("A fintech platform automating complex derivatives pricing and risk management.","exceptional"),
("An AI platform for automated crop disease detection from drone imagery.","exceptional"),
("A SaaS providing real-time grid optimization for renewable energy integration.","exceptional"),
# More WEAK to balance
("A platform for rating how soft different clouds look.","weak"),
("An app that tells you which direction you're facing but only in pig latin.","weak"),
("A subscription box of random rubber bands.","weak"),
("A SaaS that counts how many tabs you have open and judges you.","weak"),
("A marketplace for trading used birthday candles.","weak"),
("An AI that generates compliments for your furniture.","weak"),
("A platform for competitive thumb wrestling leagues.","weak"),
("An app that plays elevator music whenever you're in an elevator.","weak"),
("A smart sock that tells you when it has a hole.","weak"),
("A subscription service for monthly deliveries of different types of tape.","weak"),
("A platform for reviewing and rating different brands of paper clips.","weak"),
("An AI that predicts which of your friends will text you next.","weak"),
("A marketplace for trading collectible napkins.","weak"),
("An app that converts your walking pace into a musical tempo.","weak"),
("A SaaS that tracks how many times you say 'um' in meetings.","weak"),
("A platform for competitive staring contests via video call.","weak"),
("An app that rates the aesthetic quality of parking lots.","weak"),
("A subscription box of miniature flags from random countries.","weak"),
("A smart coaster that tells you the temperature of your drink.","weak"),
("An AI that writes haiku about your daily commute.","weak"),
]

# Load existing
full = Path(__file__).parent.parent / "data" / "training_data_full.json"
with open(full) as f:
    all_ex = json.load(f)
seen = {ex.get("source_id") for ex in all_ex}

for i, (desc, tier) in enumerate(MORE):
    eid = f"m{i+1:03d}"
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
