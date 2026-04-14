"""Build 500-example training dataset. Run: .venv/bin/python scripts/build_dataset.py"""
import json, random
from pathlib import Path
random.seed(42)

SYSTEM_MSG = ("You are a startup and project evaluator. Given a project description, "
    "evaluate it across 10 business-viability dimensions, each scored 1-10 "
    "with a brief justification. Output valid JSON.")
DIMS = ["pain_severity","pain_frequency","existing_alternatives","willingness_to_pay",
    "market_size","scalability","profitability_potential","defensibility",
    "time_to_value","founder_market_fit_requirement"]
W = {"pain_severity":.15,"pain_frequency":.10,"existing_alternatives":.10,
    "willingness_to_pay":.12,"market_size":.10,"scalability":.08,
    "profitability_potential":.12,"defensibility":.08,"time_to_value":.07,
    "founder_market_fit_requirement":.08}

REASONS = {d: {
    "weak": [], "below_avg": [], "average": [], "strong": [], "exceptional": []
} for d in DIMS}

REASONS["pain_severity"]["weak"] = ["No real problem being solved","Pure novelty, not a pain point","Creates friction rather than solving it","Addresses a non-existent need","Trivial inconvenience at best"]
REASONS["pain_severity"]["below_avg"] = ["Mild inconvenience but not critical","Nice-to-have, not must-have","Problem exists but is low priority","Hobby-level need, not urgent","Some frustration but manageable"]
REASONS["pain_severity"]["average"] = ["Moderate pain that affects productivity","Real problem but not top priority","Meaningful friction in current workflows","Costs time and money but workarounds exist","Growing pain as the market matures"]
REASONS["pain_severity"]["strong"] = ["Significant pain that directly impacts revenue","Critical workflow bottleneck for target users","Costly problem that compounds over time","Regulatory or compliance risk creates urgency","Direct impact on business outcomes"]
REASONS["pain_severity"]["exceptional"] = ["Mission-critical problem with severe consequences","Existential risk for businesses without a solution","Multi-billion dollar problem affecting entire industries","Life-or-death implications in healthcare or safety","Regulatory mandate creates forced adoption"]

REASONS["pain_frequency"]["weak"] = ["Rarely or never encountered","One-time novelty","No recurring need","Occasional at best","Seasonal or event-based only"]
REASONS["pain_frequency"]["below_avg"] = ["A few times per year","Occasional but not regular","Seasonal need","Triggered by specific events","Monthly at most"]
REASONS["pain_frequency"]["average"] = ["Weekly occurrence for target users","Regular part of business operations","Multiple times per month","Ongoing but not daily","Cyclical with business rhythms"]
REASONS["pain_frequency"]["strong"] = ["Daily workflow requirement","Multiple times per day for power users","Continuous monitoring needed","Every transaction triggers this need","Constant operational requirement"]
REASONS["pain_frequency"]["exceptional"] = ["Real-time, always-on requirement","Every second matters in this domain","Continuous 24/7 monitoring essential","Every interaction requires this","Mission-critical uptime requirement"]

REASONS["existing_alternatives"]["weak"] = ["Dozens of identical solutions exist","Completely saturated market","Free alternatives are excellent","Built-in OS features cover this","No differentiation possible"]
REASONS["existing_alternatives"]["below_avg"] = ["Several good alternatives exist","Established players serve this well","Free tools cover 80% of the need","Low switching costs from current solutions","Crowded market with clear leaders"]
REASONS["existing_alternatives"]["average"] = ["Alternatives exist but have significant gaps","Current solutions are expensive or complex","Market is served but not well","Incumbents are slow to innovate","Partial solutions exist but nothing end-to-end"]
REASONS["existing_alternatives"]["strong"] = ["Few direct competitors with major limitations","Existing solutions are outdated or overpriced","No purpose-built solution for this segment","Current approaches are manual and error-prone","Incumbents focused on enterprise, leaving SMB underserved"]
REASONS["existing_alternatives"]["exceptional"] = ["No viable solution exists today","Current approaches are fundamentally broken","Greenfield market with no direct competitors","Existing tools cannot handle the scale required","Regulatory changes created a new category"]

REASONS["willingness_to_pay"]["weak"] = ["Users expect this to be free","No perceived value worth paying for","Free alternatives are good enough","Target audience has no budget","Novelty doesn't justify payment"]
REASONS["willingness_to_pay"]["below_avg"] = ["Some might pay a small amount","Price-sensitive audience","Freemium might work but conversion low","Willingness to pay is uncertain","Low perceived value vs alternatives"]
REASONS["willingness_to_pay"]["average"] = ["Moderate willingness to pay for clear value","B2B buyers have budgets for this","Subscription viable at $20-100/month","Clear ROI justifies the cost","Competitive pricing pressure limits premium"]
REASONS["willingness_to_pay"]["strong"] = ["Strong willingness to pay for proven ROI","Enterprise budgets allocated for this category","High-value problem justifies premium pricing","Customers paying more for inferior solutions","Clear cost savings justifies price"]
REASONS["willingness_to_pay"]["exceptional"] = ["Customers eagerly pay premium for best-in-class","Mission-critical spending with large budgets","Regulatory compliance makes this a must-buy","ROI is 10x+ the cost","Customers would pay significantly more"]

REASONS["market_size"]["weak"] = ["No meaningful addressable market","Fewer than 1000 potential users","Market is shrinking","Too narrow to sustain a business","Addressable segment is negligible"]
REASONS["market_size"]["below_avg"] = ["Small niche market","Limited to specific geography","Market exists but is small","Growing slowly","Addressable market under $100M"]
REASONS["market_size"]["average"] = ["Meaningful market with room for multiple players","Growing market driven by secular trends","Addressable market $100M-$1B range","Large potential but requires market education","Expanding as the industry digitizes"]
REASONS["market_size"]["strong"] = ["Large addressable market with strong growth","Multi-billion dollar TAM","Global market with cross-border potential","Expanding rapidly due to regulatory shifts","Multiple customer segments to expand into"]
REASONS["market_size"]["exceptional"] = ["Massive global market in tens of billions","Every company in target segment is a customer","Market growing 30%+ annually","Platform opportunity with network effects","Foundational infrastructure every business needs"]

REASONS["scalability"]["weak"] = ["Physical product that doesn't scale","Requires linear headcount growth","Geographic constraints limit expansion","High marginal cost per customer","Cannot serve more without proportional cost"]
REASONS["scalability"]["below_avg"] = ["Some scalability but significant overhead","Marketplace requires building both sides","Hardware limits pure software scaling","Service delivery requires trained professionals","Moderate marginal costs"]
REASONS["scalability"]["average"] = ["SaaS model with reasonable unit economics","Platform scales but needs ongoing investment","Moderate infrastructure costs grow sub-linearly","Can expand with some customization","Scalable core with some manual processes"]
REASONS["scalability"]["strong"] = ["Pure software with near-zero marginal cost","API-based model scales horizontally","Self-serve onboarding reduces sales costs","Network effects improve product at scale","Cloud-native handles growth efficiently"]
REASONS["scalability"]["exceptional"] = ["Platform with strong network effects and viral growth","Infrastructure becomes more valuable with scale","Zero marginal cost with usage-based pricing","Flywheel where more data improves the product","Winner-take-most dynamics"]

REASONS["profitability_potential"]["weak"] = ["No viable monetization path","Negative unit economics","Free market with no willingness to pay","Costs exceed any possible revenue","Race-to-bottom pricing"]
REASONS["profitability_potential"]["below_avg"] = ["Thin margins on physical goods","High CAC relative to LTV","Competitive pricing limits margins","Revenue possible but profitability uncertain","Low average contract value"]
REASONS["profitability_potential"]["average"] = ["Viable SaaS margins of 60-70%","Clear path to profitability with scale","Multiple revenue streams possible","Moderate customer acquisition costs","Healthy unit economics past initial investment"]
REASONS["profitability_potential"]["strong"] = ["High-margin SaaS with 75%+ gross margins","Strong unit economics with LTV/CAC > 3x","Enterprise pricing with large contracts","Expansion revenue from existing customers","Platform fees on high-volume transactions"]
REASONS["profitability_potential"]["exceptional"] = ["90%+ gross margins at scale","Network effects reduce CAC over time","Pricing power from mission-critical positioning","Multiple monetization vectors","Winner-take-most with increasing returns"]

REASONS["defensibility"]["weak"] = ["Trivially copyable with no barriers","No proprietary technology or data","Feature, not a product","Any competitor can replicate in weeks","No switching costs"]
REASONS["defensibility"]["below_avg"] = ["Some brand loyalty but low switching costs","First-mover advantage is temporary","Open-source alternatives could emerge","Moderate complexity but replicable","Community provides some moat"]
REASONS["defensibility"]["average"] = ["Integration depth creates switching costs","Proprietary data provides advantage","Regulatory expertise creates barriers","Partnership network hard to replicate","Domain expertise embedded in product"]
REASONS["defensibility"]["strong"] = ["Strong network effects growing with usage","Proprietary data moat compounds over time","Deep integrations make switching costly","Regulatory approvals create barriers","Platform ecosystem with third-party devs"]
REASONS["defensibility"]["exceptional"] = ["Winner-take-most network effects","Proprietary data impossible to replicate","Multi-year regulatory approval for competitors","Deep platform lock-in with ecosystem","Becomes industry standard infrastructure"]

REASONS["time_to_value"]["weak"] = ["No value delivered","Months before any benefit","Requires fundamental behavior change","Complex onboarding with uncertain payoff","Value proposition is unclear"]
REASONS["time_to_value"]["below_avg"] = ["Weeks to see initial value","Requires significant setup","Learning curve before gains","Need critical mass before useful","Gradual value accumulation"]
REASONS["time_to_value"]["average"] = ["Days to initial value with guided onboarding","Quick setup with templates","Value visible within first week","Self-serve trial demonstrates value","Immediate partial value, full over time"]
REASONS["time_to_value"]["strong"] = ["Hours to first value with streamlined onboarding","Instant demo shows clear before/after","Plug-and-play integration","Free trial converts quickly","Immediate time or cost savings"]
REASONS["time_to_value"]["exceptional"] = ["Minutes to first value, zero configuration","Instant ROI on first interaction","Self-serve signup to value in under an hour","Viral adoption because value is obvious","Replaces manual process with instant automation"]

REASONS["founder_market_fit_requirement"]["weak"] = ["No specialized knowledge needed","Anyone can build this","Basic programming sufficient","No domain expertise required","General consumer product"]
REASONS["founder_market_fit_requirement"]["below_avg"] = ["Some industry familiarity helpful","Basic market understanding needed","General business knowledge sufficient","Light technical skills required","Community management experience helpful"]
REASONS["founder_market_fit_requirement"]["average"] = ["Meaningful domain expertise needed","Industry connections accelerate growth","Technical depth in specific area required","Regulatory landscape understanding important","Sales experience in vertical valuable"]
REASONS["founder_market_fit_requirement"]["strong"] = ["Deep domain expertise required for credibility","Industry relationships critical for distribution","Specialized AI/ML skills needed","Regulatory knowledge essential","Years of industry experience needed"]
REASONS["founder_market_fit_requirement"]["exceptional"] = ["World-class expertise in specialized field","PhD-level technical depth needed","Regulatory expertise across jurisdictions","Deep relationships with key stakeholders","Rare combination of technical and domain skills"]


SUMMARIES = {
    "weak": ["No viable business. Solves no real problem with no monetization path.","Novelty concept with zero commercial potential.","Solution looking for a problem. Target market doesn't exist.","Fundamentally flawed. No pain, no willingness to pay, no defensibility.","Would not recommend pursuing. All market signals are negative.","Interesting thought experiment but fails on every business dimension.","The core premise doesn't hold up. No evidence of real demand.","Too niche, trivial, or absurd to build a sustainable business."],
    "below_avg": ["Has a kernel of a real idea but execution challenges are significant.","Could work as a lifestyle business but unlikely to achieve venture scale.","Real but small market with established alternatives.","Interesting concept but economics don't work at scale.","Addresses a real need but willingness to pay is uncertain.","Might find a small audience but growth potential is limited.","The idea has merit but timing or competition works against it.","Could succeed in a specific niche but broader expansion difficult."],
    "average": ["Solid idea with clear market need. Execution will determine success.","Viable business with moderate growth potential. Competition is main risk.","Good product-market fit potential but needs strong execution.","Real problem with willing buyers. Challenge is building defensibility.","Promising opportunity in a growing market. Multiple monetization paths.","Strong fundamentals but faces competition from incumbents and startups.","The market is there and pain is real. Success depends on go-to-market.","Decent opportunity with clear unit economics. Scale depends on sales."],
    "strong": ["Strong opportunity with clear pain, large market, and viable model.","High-potential idea with strong unit economics and growth vectors.","Addresses critical need with high willingness to pay and data moat.","Excellent product-market fit indicators. Large and growing market.","Strong business fundamentals with clear path to profitability.","Compelling value proposition with measurable ROI. Enterprise-ready.","Well-positioned in growing market with strong tailwinds.","High-conviction opportunity. Pain, market size, and timing align."],
    "exceptional": ["Exceptional opportunity at intersection of massive need and technology.","Category-defining potential. Multi-billion dollar problem, better approach.","Rare combination of enormous market, critical pain, strong defensibility.","Platform opportunity with winner-take-most dynamics.","Transformative potential. Strong technical moat and regulatory barriers.","Best-in-class opportunity. Every dimension scores highly.","Generational opportunity to build critical industry infrastructure.","Exceptional across all dimensions. Question is execution speed."],
}

RANGES = {"weak":(1,3),"below_avg":(3,5),"average":(5,7),"strong":(7,9),"exceptional":(8,10)}

def gen_scores(tier):
    lo, hi = RANGES[tier]
    sr = {}
    for d in DIMS:
        base = random.randint(lo, hi)
        if random.random() < 0.2:
            base = max(1, min(10, base + random.choice([-2,-1,1,2])))
        if d in ("scalability","time_to_value") and tier == "weak":
            base = random.randint(1, 9)
        if d == "founder_market_fit_requirement" and tier == "exceptional":
            base = random.randint(7, 10)
        sr[d] = (base, random.choice(REASONS[d][tier]))
    return sr

def mk(id, desc, sr, summary, src="synthetic"):
    ev = {}
    for d in DIMS:
        ev[d] = {"score": sr[d][0], "reason": sr[d][1]}
    ev["overall_score"] = round(sum(sr[d][0]*W[d] for d in DIMS), 1)
    ev["summary"] = summary
    return {"source_id":id,"source_type":src,"messages":[
        {"role":"system","content":SYSTEM_MSG},
        {"role":"user","content":f"Evaluate this project: {desc}"},
        {"role":"assistant","content":json.dumps(ev,ensure_ascii=False)}]}


# (description, tier)
P = [
# WEAK ~125
("A website that tells you what day of the week it is.","weak"),("An app that plays a random fart sound when you shake your phone.","weak"),("A social media platform exclusively for left-handed people.","weak"),("A physical store selling ice to people in Antarctica.","weak"),("A subscription mailing a single grain of rice monthly.","weak"),("An AI generating random numbers and claiming they are lucky.","weak"),("A dating app where you only communicate through interpretive dance videos.","weak"),("A blockchain platform for trading virtual pet rocks.","weak"),("A mobile app counting how many times you blink per day.","weak"),("A consulting firm advising on optimal office temperature.","weak"),
("A VR experience simulating waiting in line at the DMV.","weak"),("A newsletter summarizing yesterday's weather.","weak"),("A premium bottled water brand sourced from public fountains.","weak"),("An app translating cat meows into English using AI.","weak"),("A coworking space exclusively for astrologers.","weak"),("A Chrome extension replacing all text with Comic Sans.","weak"),("A meal delivery service delivering only room-temperature food.","weak"),("A fitness app tracking only how many times you sit down.","weak"),("A luxury concierge for houseplant playdates.","weak"),("A podcast that reads the phone book out loud.","weak"),
("A Tinder for matching people with parking spots.","weak"),("An NFT collection of AI-generated toast pictures.","weak"),("A SaaS adding CONFIDENTIAL watermarks to every email.","weak"),("A wearable vibrating when someone nearby sneezes.","weak"),("A marketplace for used chewing gum.","weak"),("An app rating chair comfort using phone sensors.","weak"),("A social network where you can only post during full moons.","weak"),("A delivery service for single ice cubes.","weak"),("An AI assistant responding only in haiku.","weak"),("A subscription box of 1990s office supplies.","weak"),
("A browser extension adding googly eyes to every face online.","weak"),("A platform for competitive yawning tournaments.","weak"),("An app counting steps to the nearest bathroom.","weak"),("A B2B service writing passive-aggressive emails for managers.","weak"),("A smart doorbell that only works on Tuesdays.","weak"),("A language app teaching only extinct languages.","weak"),("A ride-sharing service for people wearing hats.","weak"),("An AI predicting what your pet dreams about.","weak"),("A fintech app rounding down purchases and keeping the difference.","weak"),("A job board for positions paying only in crypto.","weak"),
("A smart mirror criticizing your outfit.","weak"),("A meditation app playing construction noise.","weak"),("A food delivery app randomly changing your order.","weak"),("A CRM storing only complaints, not contacts.","weak"),("A weather app showing weather for cities you've never visited.","weak"),("A platform renting umbrellas by the minute.","weak"),("An AI making emails exactly 17% longer.","weak"),("A fitness tracker counting only calories burned while sleeping.","weak"),("A social app requiring vegetable profile pictures.","weak"),("A travel site for trips under 5 miles.","weak"),
("A music service playing songs in reverse.","weak"),("A project tool assigning tasks randomly.","weak"),("A dating app matching by blood type.","weak"),("An e-commerce store selling artisanal dirt.","weak"),("A video call tool where everyone appears as a potato.","weak"),("A smart water bottle insulting you for not drinking.","weak"),("An app converting texts to Morse code before sending.","weak"),("A subscription for monthly bubble wrap deliveries.","weak"),("A platform for watching paint dry together.","weak"),("An AI generating excuses for being late.","weak"),
("A marketplace for celebrity-shaped soap.","weak"),("A browser extension replacing images with Nicolas Cage.","weak"),("A SaaS scheduling meetings at worst times.","weak"),("A wearable tracking doors opened per day.","weak"),("A food app recommending only restaurants you've visited.","weak"),("A cloud storage with 1 MB max capacity.","weak"),("An app notifying you of every celebrity tweet.","weak"),("A platform for competitive rock-paper-scissors.","weak"),("A smart pen writing only in invisible ink.","weak"),("A subscription box of keys that open nothing.","weak"),
("A task app where completed tasks reappear daily.","weak"),("An AI chatbot answering questions with questions.","weak"),("A ride-share exclusively for grocery trips.","weak"),("A social platform deleting posts after 3 seconds.","weak"),("A fintech converting savings into pennies.","weak"),("A smart home device randomly toggling lights.","weak"),("A language app teaching pirate speak.","weak"),("A job board for unpaid internships at unlaunched startups.","weak"),("A meal kit with all dehydrated ingredients.","weak"),("A VR game simulating tax filing.","weak"),
("A platform rating restrooms by decor only.","weak"),("An AI predicting weather from your mood.","weak"),("A marketplace for expired coupons.","weak"),("A fitness app working only when standing still.","weak"),("A smart fridge ordering food you're allergic to.","weak"),("A social network for people named Dave.","weak"),("A SaaS converting spreadsheets to dance instructions.","weak"),("A travel app suggesting worst-weather destinations.","weak"),("A podcast of office equipment ASMR.","weak"),("An app measuring banana curvature with AR.","weak"),
("A B2B platform outsourcing high-fives at events.","weak"),("A ring glowing when Mercury is in retrograde.","weak"),("A subscription delivering one puzzle piece monthly.","weak"),("An AI writing apology letters to houseplants.","weak"),("A dating app matching by shoe size.","weak"),("A cloud platform for storing dreams.","weak"),("A smart toothbrush posting brushing stats to social media.","weak"),("A marketplace for hand-drawn fictional maps.","weak"),("An app translating baby cries into stock predictions.","weak"),("A SaaS visualizing unread emails as abstract art.","weak"),
("A delivery service for glitter bombs.","weak"),("A fitness app counting only backward steps.","weak"),("A browser extension replacing numbers with Roman numerals.","weak"),("A platform for speed-reading terms and conditions.","weak"),("An AI generating corporate jargon from simple sentences.","weak"),
# BELOW_AVG ~100
("A subscription box of curated artisanal hot sauces.","below_avg"),("An online marketplace for used textbooks between college students.","below_avg"),("A social network for vintage typewriter collectors.","below_avg"),("A dating app for dog owners matching by dog compatibility.","below_avg"),("A wearable monitoring soil moisture for home gardeners.","below_avg"),("A Chrome extension blocking social media during work hours.","below_avg"),("A mobile game identifying bird species from audio.","below_avg"),("A blockchain protocol for verifying academic credentials.","below_avg"),("A service renting soundproof phone booths to coworking spaces.","below_avg"),("A platform booking mobile car detailing on-demand.","below_avg"),
("A browser extension summarizing Terms of Service.","below_avg"),("An app gamifying personal finance for teenagers.","below_avg"),("A smart trash can sorting recyclables with AI.","below_avg"),("An app matching hiking partners by fitness level.","below_avg"),("A platform for fractional ownership of rare sneakers.","below_avg"),("A marketplace for local artists selling custom phone cases.","below_avg"),("An app finding quiet cafes by real-time noise levels.","below_avg"),("A subscription for playlists based on astrological sign.","below_avg"),("A platform connecting amateur photographers with headshot seekers.","below_avg"),("An AI generating workout playlists from heart rate.","below_avg"),
("A marketplace renting designer handbags for occasions.","below_avg"),("An app splitting household chores fairly among roommates.","below_avg"),("A platform booking local off-the-beaten-path tour guides.","below_avg"),("A SaaS managing community garden plots.","below_avg"),("An app connecting pet owners with pet-friendly businesses.","below_avg"),("A platform for indie board game playtesting with remote players.","below_avg"),("A subscription delivering monthly craft beer from microbreweries.","below_avg"),("An AI suggesting outfit combinations from your wardrobe.","below_avg"),("A platform connecting retired teachers with students for tutoring.","below_avg"),("A smart plant pot with automatic watering and light adjustment.","below_avg"),
("An app tracking and trading loyalty points across stores.","below_avg"),("A marketplace for commissioning custom illustrations.","below_avg"),("A platform for neighborhood tool sharing.","below_avg"),("An app finding and joining local sports pickup games.","below_avg"),("A SaaS for small churches managing donations and events.","below_avg"),("A platform connecting home cooks with neighbors wanting meals.","below_avg"),("An app tracking caffeine intake and optimal coffee timing.","below_avg"),("A marketplace for vintage video game cartridges.","below_avg"),("A platform booking mobile massage therapists.","below_avg"),("An AI generating personalized bedtime stories for children.","below_avg"),
("A SaaS managing shared vacation home schedules.","below_avg"),("A platform for language practice over coffee.","below_avg"),("An app cataloging personal book collections.","below_avg"),("A marketplace for farmers selling directly via weekly boxes.","below_avg"),("A platform connecting dog walkers with local dog owners.","below_avg"),("An app finding free parking with crowdsourced data.","below_avg"),("A SaaS for yoga studios managing schedules and bookings.","below_avg"),("A platform renting camera equipment from photographers.","below_avg"),("An AI journaling app with weekly mental health insights.","below_avg"),("A marketplace for handmade pet accessories.","below_avg"),
]

P += [
# AVERAGE ~100
("A SaaS auto-generating API documentation from source code.","average"),("A platform comparing insurance quotes for gig workers.","average"),("A telehealth platform for mental health in rural areas.","average"),("A platform connecting retired professionals with startups for advisory.","average"),("An app scanning receipts to track nutritional intake with CV.","average"),("A drone service inspecting solar panel installations.","average"),("A platform connecting homeowners with vetted contractors for small repairs.","average"),("An AI resume screening tool removing bias indicators.","average"),("A vertical SaaS for managing veterinary clinics.","average"),("A platform for musicians to license music to content creators.","average"),
("An enterprise tool monitoring employee burnout via communication patterns.","average"),("A SaaS managing influencer marketing campaigns end-to-end.","average"),("A subscription delivering meal kits for specific medical diets.","average"),("An AI writing assistant for STEM research papers.","average"),("A no-code platform for interactive SaaS product demos.","average"),("A platform helping small businesses manage online reviews with AI.","average"),("An API converting PDFs into structured queryable data.","average"),("A marketplace for fractional commercial real estate ownership.","average"),("A P2P lending platform for small businesses in emerging markets.","average"),("An AI copilot for real estate agents automating listings and follow-up.","average"),
("A B2B platform connecting restaurants with local farms.","average"),("A SaaS detecting duplicate customer records across CRMs.","average"),("A platform for corporate teams to book offsite retreats.","average"),("An AI legal document review tool for small law firms.","average"),("A SaaS for property managers handling maintenance and tenant comms.","average"),("A marketplace connecting freelance CFOs with startups.","average"),("An AI monitoring brand mentions and generating sentiment reports.","average"),("A platform for e-commerce brands managing returns.","average"),("A SaaS for restaurants managing inventory and reducing waste.","average"),("An AI customer support chatbot builder for small businesses.","average"),
("A platform connecting translators with businesses.","average"),("A SaaS managing employee onboarding and compliance training.","average"),("An AI generating product descriptions from photos.","average"),("A platform for corporate catering from local restaurants.","average"),("A SaaS for nonprofits managing donors and grants.","average"),("An AI code review tool catching security vulnerabilities.","average"),("A platform connecting physical therapists with remote patients.","average"),("A SaaS managing construction project timelines and budgets.","average"),("An AI generating social media content calendars.","average"),("A platform for manufacturers finding raw material suppliers.","average"),
("A SaaS managing co-living spaces.","average"),("An AI transcribing and summarizing sales calls.","average"),("A platform connecting pharmacies with wholesale distributors.","average"),("A SaaS for event planners managing vendors and budgets.","average"),("An AI generating personalized sales email sequences.","average"),("A platform for small businesses pooling purchasing power.","average"),("A SaaS managing franchise operations across locations.","average"),("An AI detecting and preventing ad fraud.","average"),("A platform connecting insurance agents with carriers.","average"),("A SaaS managing clinical trials and patient recruitment.","average"),
# STRONG ~125
("An AI platform automating patent application generation and filing.","strong"),("A SaaS automating accounts payable for mid-market companies.","strong"),("An AI cybersecurity platform detecting threats in real-time for SMBs.","strong"),("A vertical SaaS for dental practices managing all operations.","strong"),("A platform automating compliance monitoring for financial institutions.","strong"),("An AI supply chain optimization platform for manufacturers.","strong"),("A developer platform providing auth, payments, and notifications as APIs.","strong"),("A SaaS managing commercial fleet vehicles end-to-end.","strong"),("An AI medical imaging tool assisting radiologists.","strong"),("A platform automating multi-state tax prep for small businesses.","strong"),
("An AI recruiting platform matching candidates by skills not resumes.","strong"),("A SaaS managing subscription billing and churn analytics.","strong"),("An AI optimizing cloud infrastructure costs across providers.","strong"),("A platform providing embedded B2B lending and BNPL.","strong"),("A SaaS managing multi-location retail operations.","strong"),("An AI contract analysis tool extracting terms and flagging risks.","strong"),("A platform automating insurance claims with AI document analysis.","strong"),("A SaaS managing clinical workflows and EHR for specialty clinics.","strong"),("An AI fraud detection platform for online payments.","strong"),("A developer tool auto-generating database migration scripts.","strong"),
("A SaaS managing warehouse picking, packing, and shipping.","strong"),("An AI personalizing e-commerce recommendations from browsing data.","strong"),("A platform providing real-time translation for customer support.","strong"),("A SaaS optimizing digital ad spend across platforms.","strong"),("An AI predictive maintenance platform for industrial equipment.","strong"),("A platform automating B2B accounts receivable and collections.","strong"),("A SaaS managing food safety compliance for restaurant chains.","strong"),("An AI generating and optimizing landing pages from conversion data.","strong"),("A platform providing embedded insurance for e-commerce and gig platforms.","strong"),("A SaaS automating enterprise procurement workflows.","strong"),
("A developer platform providing serverless database infrastructure.","strong"),("An AI monitoring regulatory changes and alerting businesses.","strong"),("A SaaS managing loyalty programs across retail chains.","strong"),("An AI automating financial modeling for CFOs.","strong"),("A platform providing KYC and identity verification via API.","strong"),("A SaaS optimizing last-mile delivery operations.","strong"),("An AI generating video content from text scripts.","strong"),("A platform automating global payroll and tax compliance.","strong"),("A SaaS managing energy consumption and sustainability reporting.","strong"),("An AI detecting account takeover fraud for online platforms.","strong"),
("A platform providing embedded fintech infrastructure for SaaS companies.","strong"),("A SaaS managing clinical documentation and coding for healthcare.","strong"),("An AI optimizing pricing dynamically based on demand and competition.","strong"),("A platform automating vendor risk and compliance monitoring.","strong"),("A developer tool for real-time error monitoring and debugging.","strong"),("A SaaS automating employee benefits administration.","strong"),("An AI generating architectural floor plans from descriptions.","strong"),("A platform providing managed data integration and ETL pipelines.","strong"),("A SaaS managing field service operations for utilities.","strong"),("An AI automating M&A due diligence document review.","strong"),
("A platform providing real-time freight rate comparison and booking.","strong"),("A SaaS for managing and optimizing cold chain logistics.","strong"),("An AI-powered platform for automated medical coding and billing.","strong"),("A developer tool providing automated API testing and monitoring.","strong"),("A SaaS managing multi-channel e-commerce inventory and orders.","strong"),
# EXCEPTIONAL ~50
("An AI automating end-to-end drug discovery.","exceptional"),("A globally distributed edge computing platform with sub-10ms latency.","exceptional"),("An AI cybersecurity platform autonomously detecting and remediating threats.","exceptional"),("A fintech platform for real-time cross-border B2B payments with FX hedging.","exceptional"),("An AI platform enabling custom foundation model deployment on proprietary data.","exceptional"),("A vertical SaaS digitizing commercial insurance underwriting.","exceptional"),("An AI generating production-ready mobile apps from natural language.","exceptional"),("A platform providing embedded banking infrastructure via API.","exceptional"),("An AI drug repurposing platform finding new uses for approved drugs.","exceptional"),("A developer platform for automated cloud security scanning and remediation.","exceptional"),
("An AI automating financial audits for public companies.","exceptional"),("A SaaS providing real-time global supply chain visibility and disruption alerts.","exceptional"),("An AI platform for autonomous vehicle fleet management.","exceptional"),("A platform providing programmable telecom infrastructure via API.","exceptional"),("An AI clinical decision support system for diagnosis.","exceptional"),("A fintech automating treasury management and cash forecasting.","exceptional"),("An AI generating synthetic training data for ML models.","exceptional"),("A SaaS for enterprise carbon accounting and offset management.","exceptional"),("An AI automating the entire mortgage origination process.","exceptional"),("A developer platform for unified observability with AI root cause analysis.","exceptional"),
("An AI enabling real-time video call translation with lip-sync.","exceptional"),("A SaaS automating commercial real estate transactions.","exceptional"),("An AI platform for personalized cancer treatment from genomic analysis.","exceptional"),("A fintech providing instant business credit from real-time accounting data.","exceptional"),("A developer tool for AI-powered database query optimization.","exceptional"),("An AI automating regulatory compliance for global banks.","exceptional"),("A SaaS managing renewable energy assets including solar and battery.","exceptional"),("An AI detecting money laundering across financial networks in real-time.","exceptional"),("A platform providing programmable payment infrastructure for marketplaces.","exceptional"),("An AI automating clinical trial matching for oncology patients.","exceptional"),
("An AI platform automating pharmaceutical manufacturing quality control.","exceptional"),("A developer platform providing AI-powered infrastructure-as-code generation.","exceptional"),("A fintech platform enabling instant settlement for securities trading.","exceptional"),("An AI automating satellite imagery analysis for agriculture and defense.","exceptional"),("A SaaS providing real-time fraud detection across banking channels.","exceptional"),("An AI platform for autonomous robotic surgery assistance.","exceptional"),("A developer tool providing AI-powered code migration across languages.","exceptional"),("A fintech platform automating cross-border trade finance and letters of credit.","exceptional"),("An AI platform for real-time protein structure prediction and drug design.","exceptional"),("A SaaS automating end-to-end healthcare revenue cycle management.","exceptional"),
("An AI platform for autonomous drone inspection of critical infrastructure.","exceptional"),("A developer platform providing zero-trust security mesh for microservices.","exceptional"),("A fintech enabling programmable money with smart contract-based payments.","exceptional"),("An AI automating radiology report generation from medical imaging.","exceptional"),("A SaaS providing AI-powered demand forecasting for retail supply chains.","exceptional"),("An AI platform for real-time natural disaster prediction and response.","exceptional"),("A developer tool providing AI-assisted formal verification of software.","exceptional"),("A fintech platform automating insurance pricing with real-time risk models.","exceptional"),("An AI platform for personalized drug dosing based on pharmacogenomics.","exceptional"),("A SaaS automating government procurement and contract management.","exceptional"),
]


# Edge cases with hand-crafted scores
EDGE = [
    mk("e001","an app for food",
       {d:(3,random.choice(REASONS[d]["below_avg"])) for d in DIMS},
       "Description too vague to evaluate. Need specifics on problem, audience, and solution."),
    mk("e002","We're building the Uber for dogs. It's like Airbnb meets Tinder but for pets. We use blockchain, AI, and quantum computing.",
       {d:(3,random.choice(REASONS[d]["below_avg"])) for d in DIMS},
       "Buzzword overload. No clear problem or solution articulated. Pet industry is real but this pitch is unfocused."),
    mk("e003","asdfghjkl qwerty zxcvbn",
       {d:(1,"Not a coherent project description — cannot evaluate") for d in DIMS},
       "Invalid input. Not a project description."),
    mk("e004","A platform that helps people find and hire hitmen.",
       {d:(1,"Illegal activity — not a legitimate business") for d in DIMS},
       "Describes illegal activity. Cannot be evaluated as a business."),
    mk("e005","What is the meaning of life?",
       {d:(1,"Philosophical question, not a project description") for d in DIMS},
       "Not a project description. Please provide a product or business idea."),
    mk("e006",".",
       {d:(1,"Empty input — cannot evaluate") for d in DIMS},
       "Input too short to evaluate."),
    mk("e007","I want to build something with AI that makes money. Not sure what yet.",
       {d:(2,random.choice(REASONS[d]["weak"])) for d in DIMS},
       "Too vague. 'Something with AI' is not a business idea. Needs specific problem and solution."),
    mk("e008","Una aplicación que ayuda a restaurantes pequeños a gestionar pedidos y reducir desperdicio de alimentos en un 30%.",
       gen_scores("average"),
       "Strong idea with clear ROI. 30% waste reduction needs validation. Competition exists but small restaurants underserved. (Note: Spanish input.)"),
]

def main():
    existing = Path(__file__).parent.parent / "data" / "training_data.json"
    with open(existing) as f:
        base = json.load(f)
    seen = {ex.get("source_id") for ex in base}
    all_ex = list(base)

    # Add generated examples
    for i, (desc, tier) in enumerate(P):
        eid = f"g{i+1:03d}"
        if eid not in seen:
            sr = gen_scores(tier)
            all_ex.append(mk(eid, desc, sr, random.choice(SUMMARIES[tier])))
            seen.add(eid)

    # Add edge cases
    for e in EDGE:
        if e["source_id"] not in seen:
            all_ex.append(e)
            seen.add(e["source_id"])

    random.shuffle(all_ex)

    # Stats
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

    out = Path(__file__).parent.parent / "data" / "training_data_full.json"
    with open(out,"w") as f:
        json.dump(all_ex, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {out}")

if __name__ == "__main__":
    main()
