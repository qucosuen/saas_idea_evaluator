"""
Build Full Training Dataset (500 examples)
============================================
Programmatically generates a score-calibrated training dataset.

Usage:
  .venv/bin/python scripts/build_training_dataset.py
"""

import json
import random
from pathlib import Path

random.seed(42)

SYSTEM_MSG = (
    "You are a startup and project evaluator. Given a project description, "
    "evaluate it across 10 business-viability dimensions, each scored 1-10 "
    "with a brief justification. Output valid JSON."
)

DIMS = [
    "pain_severity", "pain_frequency", "existing_alternatives",
    "willingness_to_pay", "market_size", "scalability",
    "profitability_potential", "defensibility", "time_to_value",
    "founder_market_fit_requirement"
]

WEIGHTS = {
    "pain_severity": 0.15, "pain_frequency": 0.10,
    "existing_alternatives": 0.10, "willingness_to_pay": 0.12,
    "market_size": 0.10, "scalability": 0.08,
    "profitability_potential": 0.12, "defensibility": 0.08,
    "time_to_value": 0.07, "founder_market_fit_requirement": 0.08,
}

def make_example(id_str, desc, sr, summary, src="synthetic"):
    ev = {}
    for d in DIMS:
        ev[d] = {"score": sr[d][0], "reason": sr[d][1]}
    ev["overall_score"] = round(sum(sr[d][0]*WEIGHTS[d] for d in DIMS), 1)
    ev["summary"] = summary
    return {
        "source_id": id_str, "source_type": src,
        "messages": [
            {"role": "system", "content": SYSTEM_MSG},
            {"role": "user", "content": f"Evaluate this project: {desc}"},
            {"role": "assistant", "content": json.dumps(ev, ensure_ascii=False)},
        ]
    }


# ============================================================================
# PROJECT TEMPLATES BY TIER
# Each entry: (description, {dim: (score, reason)}, summary)
# ============================================================================

ALL_EXAMPLES = []

# --- TIER 1: WEAK (overall ~1.5-3.5) ---
WEAK = [
("A website that tells you what day of the week it is.",
 {"pain_severity":(1,"Everyone already knows what day it is"),"pain_frequency":(1,"Not a real need"),"existing_alternatives":(1,"Every phone and computer shows this"),"willingness_to_pay":(1,"No one would pay"),"market_size":(1,"No addressable market"),"scalability":(9,"Static page scales trivially"),"profitability_potential":(1,"Zero monetization"),"defensibility":(1,"Built in 5 minutes"),"time_to_value":(10,"Instant"),"founder_market_fit_requirement":(1,"No skills needed")},
 "Joke project with no business viability whatsoever."),
("An app that plays a random fart sound when you shake your phone.",
 {"pain_severity":(1,"Pure novelty"),"pain_frequency":(2,"Occasional party use"),"existing_alternatives":(1,"Dozens of identical apps exist"),"willingness_to_pay":(1,"Free app category"),"market_size":(2,"Tiny novelty audience"),"scalability":(9,"App scales trivially"),"profitability_potential":(1,"Negligible ad revenue"),"defensibility":(1,"Copyable in an afternoon"),"time_to_value":(9,"Instant entertainment"),"founder_market_fit_requirement":(1,"Basic mobile dev")},
 "Novelty app with zero business potential. Saturated category."),
("A social media platform exclusively for left-handed people.",
 {"pain_severity":(1,"Handedness is not a social pain point"),"pain_frequency":(1,"No recurring need"),"existing_alternatives":(1,"Existing social media works for everyone"),"willingness_to_pay":(1,"No reason to pay"),"market_size":(2,"10% of population but none need this"),"scalability":(7,"Software scales"),"profitability_potential":(1,"No business model"),"defensibility":(1,"No moat"),"time_to_value":(5,"Need critical mass"),"founder_market_fit_requirement":(1,"No expertise needed")},
 "Solution without a problem. Handedness is not a meaningful social identity."),
("A physical store that only sells ice to people in Antarctica.",
 {"pain_severity":(1,"Antarctica has unlimited ice"),"pain_frequency":(1,"No one needs to buy ice there"),"existing_alternatives":(1,"Ice is everywhere in Antarctica"),"willingness_to_pay":(1,"Absurd to pay for ice there"),"market_size":(1,"~1000 researchers"),"scalability":(1,"Physical store cannot scale"),"profitability_potential":(1,"Negative unit economics"),"defensibility":(1,"No advantage"),"time_to_value":(1,"No value to deliver"),"founder_market_fit_requirement":(1,"No expertise helps")},
 "Absurd concept. Zero viability on every dimension."),
("A subscription that mails you a single grain of rice monthly.",
 {"pain_severity":(1,"No one needs this"),"pain_frequency":(1,"No recurring need"),"existing_alternatives":(1,"Rice at every grocery store"),"willingness_to_pay":(1,"Shipping exceeds product value"),"market_size":(1,"No market"),"scalability":(1,"Physical fulfillment with negative margins"),"profitability_potential":(1,"Guaranteed money loser"),"defensibility":(1,"No one would copy this"),"time_to_value":(1,"No value"),"founder_market_fit_requirement":(1,"No expertise relevant")},
 "Intentionally absurd. Negative unit economics, no market."),
("An AI that generates random numbers and claims they are lucky.",
 {"pain_severity":(1,"Superstition is not a pain point"),"pain_frequency":(2,"Some check daily"),"existing_alternatives":(1,"Random generators everywhere"),"willingness_to_pay":(1,"Free alternatives exist"),"market_size":(2,"Small superstitious audience"),"scalability":(9,"Trivial computation"),"profitability_potential":(1,"No monetization"),"defensibility":(1,"No IP"),"time_to_value":(9,"Instant"),"founder_market_fit_requirement":(1,"No skills needed")},
 "Trivial novelty. Random number generation requires no AI."),
("A dating app where you can only communicate through interpretive dance videos.",
 {"pain_severity":(2,"Creates friction rather than solving it"),"pain_frequency":(3,"Dating is ongoing"),"existing_alternatives":(1,"Video dating without restrictions exists"),"willingness_to_pay":(1,"Novelty wears off"),"market_size":(1,"Dancers who are single is tiny"),"scalability":(6,"Video platform needs infrastructure"),"profitability_potential":(1,"Too niche"),"defensibility":(1,"Gimmick easily copied"),"time_to_value":(2,"Need to learn dance first"),"founder_market_fit_requirement":(2,"Dance and tech background")},
 "Creates friction rather than solving it. The restriction is a bug, not a feature."),
("A blockchain platform for trading virtual pet rocks.",
 {"pain_severity":(1,"Virtual pet rocks solve nothing"),"pain_frequency":(1,"No recurring need"),"existing_alternatives":(2,"NFT marketplaces exist"),"willingness_to_pay":(2,"Brief speculative interest"),"market_size":(2,"Tiny speculative niche"),"scalability":(7,"Blockchain scales technically"),"profitability_potential":(2,"Low-volume transaction fees"),"defensibility":(1,"Any marketplace can add this"),"time_to_value":(5,"Need marketplace liquidity"),"founder_market_fit_requirement":(4,"Blockchain dev skills")},
 "Combines hype cycles without solving a real problem."),
("A mobile app that counts how many times you blink per day.",
 {"pain_severity":(1,"Blink counting has no practical value"),"pain_frequency":(2,"Could track daily"),"existing_alternatives":(2,"Health trackers monitor this"),"willingness_to_pay":(1,"No one pays for this"),"market_size":(1,"No market"),"scalability":(7,"App scales but drains battery"),"profitability_potential":(1,"No monetization"),"defensibility":(1,"Trivial feature"),"time_to_value":(6,"Works immediately but useless"),"founder_market_fit_requirement":(3,"Computer vision basics")},
 "Technically interesting but commercially worthless."),
("A consulting firm advising companies on optimal office temperature.",
 {"pain_severity":(2,"Office temperature complaints are minor"),"pain_frequency":(2,"Seasonal issue"),"existing_alternatives":(2,"HVAC companies handle this"),"willingness_to_pay":(2,"Won't pay consultants for thermostat advice"),"market_size":(3,"Many offices but tiny need"),"scalability":(2,"Consulting doesn't scale"),"profitability_potential":(2,"Low willingness to pay"),"defensibility":(1,"No proprietary knowledge"),"time_to_value":(4,"Need site assessment"),"founder_market_fit_requirement":(3,"HVAC knowledge")},
 "Real but trivial problem. Consulting doesn't scale, willingness to pay near zero."),
("A VR experience simulating waiting in line at the DMV.",
 {"pain_severity":(1,"No one wants to simulate unpleasant experiences"),"pain_frequency":(1,"No demand"),"existing_alternatives":(1,"Real lines are free"),"willingness_to_pay":(1,"People pay to avoid lines"),"market_size":(1,"No market"),"scalability":(7,"VR content scales"),"profitability_potential":(1,"Anti-value proposition"),"defensibility":(1,"No one would copy"),"time_to_value":(3,"Requires VR headset"),"founder_market_fit_requirement":(4,"VR dev skills")},
 "Anti-value proposition. Simulates something people actively avoid."),
("A newsletter summarizing yesterday's weather.",
 {"pain_severity":(1,"Yesterday's weather is irrelevant"),"pain_frequency":(1,"No daily need"),"existing_alternatives":(1,"Weather apps show history free"),"willingness_to_pay":(1,"Zero value"),"market_size":(1,"No market"),"scalability":(8,"Email scales"),"profitability_potential":(1,"No monetization"),"defensibility":(1,"Trivially replicable"),"time_to_value":(3,"Always a day late"),"founder_market_fit_requirement":(1,"No expertise")},
 "Information with zero utility. Forecasts are valuable; recaps are not."),
("A premium bottled water brand sourced from public drinking fountains.",
 {"pain_severity":(1,"Public water is free"),"pain_frequency":(3,"People drink daily"),"existing_alternatives":(1,"Tap water and bottled brands"),"willingness_to_pay":(1,"No premium justification"),"market_size":(2,"Huge market but no positioning"),"scalability":(2,"Physical product logistics"),"profitability_potential":(1,"Negative margins"),"defensibility":(1,"No brand story"),"time_to_value":(5,"Immediate consumption"),"founder_market_fit_requirement":(2,"Beverage basics")},
 "No differentiation. Selling free water at premium without any brand story."),
("An app translating cat meows into English using AI.",
 {"pain_severity":(2,"Pet owners curious but not a real pain"),"pain_frequency":(3,"Daily cat interaction"),"existing_alternatives":(2,"MeowTalk exists"),"willingness_to_pay":(2,"Some might pay $1-2"),"market_size":(4,"Large cat owner population"),"scalability":(7,"App scales"),"profitability_potential":(2,"Poor retention, low LTV"),"defensibility":(2,"AI model differentiable but accuracy questionable"),"time_to_value":(7,"Instant translation attempt"),"founder_market_fit_requirement":(5,"Audio ML expertise")},
 "Fun novelty with poor retention. Translation accuracy is unverifiable."),
("A coworking space exclusively for astrologers.",
 {"pain_severity":(2,"No unique workspace needs"),"pain_frequency":(3,"Daily workspace need"),"existing_alternatives":(2,"Regular coworking serves everyone"),"willingness_to_pay":(3,"Some pay for community"),"market_size":(1,"Extremely few professional astrologers"),"scalability":(1,"Physical space"),"profitability_potential":(2,"Tiny market"),"defensibility":(2,"Community loyalty minimal"),"time_to_value":(5,"Immediate access"),"founder_market_fit_requirement":(3,"Real estate management")},
 "Hyper-niche physical business. Too few astrologers to fill one location."),
("A Chrome extension replacing all text with Comic Sans.",
 {"pain_severity":(1,"No pain addressed"),"pain_frequency":(1,"Novelty wears off in minutes"),"existing_alternatives":(1,"Font extensions exist"),"willingness_to_pay":(1,"Free category"),"market_size":(1,"Tiny"),"scalability":(9,"Trivial"),"profitability_potential":(1,"No revenue"),"defensibility":(1,"10 lines of CSS"),"time_to_value":(9,"Instant"),"founder_market_fit_requirement":(1,"Basic web dev")},
 "Trivial novelty. No pain, no market, no money."),
("A meal delivery service delivering only room-temperature food.",
 {"pain_severity":(1,"People prefer hot or cold food"),"pain_frequency":(2,"Daily eating"),"existing_alternatives":(1,"Every service delivers at intended temperature"),"willingness_to_pay":(1,"Anti-feature reduces willingness"),"market_size":(1,"No one wants this"),"scalability":(2,"Physical delivery"),"profitability_potential":(1,"Negative differentiation"),"defensibility":(1,"No one would copy a bad idea"),"time_to_value":(3,"Delivery plus disappointment"),"founder_market_fit_requirement":(3,"Food service ops")},
 "Negative differentiation. The constraint makes the product worse."),
("A fitness app tracking only how many times you sit down daily.",
 {"pain_severity":(2,"Sedentary behavior concern but metric alone useless"),"pain_frequency":(5,"Sitting happens many times daily"),"existing_alternatives":(2,"Activity trackers cover this better"),"willingness_to_pay":(1,"Too narrow to pay for"),"market_size":(2,"People want comprehensive tracking"),"scalability":(7,"App scales"),"profitability_potential":(1,"Single-metric no value"),"defensibility":(1,"Any fitness app can add this"),"time_to_value":(6,"Immediate but no insight"),"founder_market_fit_requirement":(2,"Basic app dev")},
 "Too narrow to be useful. Single metric without context has no value."),
("A luxury concierge for arranging playdates between wealthy people's houseplants.",
 {"pain_severity":(1,"Plants don't need playdates"),"pain_frequency":(1,"Not a real activity"),"existing_alternatives":(1,"No demand exists"),"willingness_to_pay":(2,"Ultra-wealthy might pay for absurd things"),"market_size":(1,"No market"),"scalability":(1,"Physical service"),"profitability_potential":(1,"No viable business"),"defensibility":(1,"No moat needed"),"time_to_value":(2,"Plant logistics"),"founder_market_fit_requirement":(2,"Concierge experience")},
 "Absurd concept. Even luxury concierge needs actual human desires."),
("A podcast that reads the phone book out loud.",
 {"pain_severity":(1,"Phone books are obsolete"),"pain_frequency":(1,"No demand"),"existing_alternatives":(1,"Directories searchable online"),"willingness_to_pay":(1,"Zero value"),"market_size":(1,"No audience"),"scalability":(7,"Podcast distributes easily"),"profitability_potential":(1,"No sponsors"),"defensibility":(1,"No competition for this"),"time_to_value":(1,"Immediate boredom"),"founder_market_fit_requirement":(1,"Can read aloud")},
 "Content with negative entertainment value."),
]

for i, (d, sr, s) in enumerate(WEAK):
    ALL_EXAMPLES.append(make_example(f"w{i+1:03d}", d, sr, s))



# ============================================================================
# TEMPLATE-BASED GENERATION for remaining tiers
# ============================================================================

# Project templates: (description_template, industry, tier_hint)
# tier_hint: "weak", "below_avg", "average", "strong", "exceptional"

PROJECTS = [
# --- MORE WEAK (21-125) ---
("A Tinder for matching people with their ideal parking spot.", "Consumer", "weak"),
("An NFT collection of AI-generated pictures of toast.", "Web3", "weak"),
("A SaaS tool that adds a watermark saying 'CONFIDENTIAL' to every email you send.", "SaaS", "weak"),
("A wearable that vibrates every time someone nearby sneezes.", "Hardware", "weak"),
("A marketplace for buying and selling used chewing gum.", "Marketplace", "weak"),
("An app that rates how comfortable your current chair is using phone sensors.", "Consumer", "weak"),
("A social network where you can only post during full moons.", "Social", "weak"),
("A delivery service for single ice cubes.", "Logistics", "weak"),
("An AI assistant that only responds with haiku poems.", "AI", "weak"),
("A subscription box of random office supplies from the 1990s.", "E-commerce", "weak"),
("A browser extension that adds googly eyes to every face on the internet.", "Consumer", "weak"),
("A platform for competitive yawning tournaments.", "Entertainment", "weak"),
("An app that tells you the exact number of steps to the nearest bathroom.", "Consumer", "weak"),
("A B2B service that writes passive-aggressive emails on behalf of managers.", "SaaS", "weak"),
("A smart doorbell that only works on Tuesdays.", "Hardware", "weak"),
("A language learning app that only teaches extinct languages.", "EdTech", "weak"),
("A ride-sharing service exclusively for people wearing hats.", "Mobility", "weak"),
("An AI tool that predicts what your pet is dreaming about.", "AI/Consumer", "weak"),
("A fintech app that rounds down your purchases and keeps the difference.", "Fintech", "weak"),
("A job board exclusively for positions that pay in cryptocurrency.", "HR/Web3", "weak"),
("A smart mirror that criticizes your outfit choices.", "Hardware", "weak"),
("A meditation app that plays construction noise.", "Wellness", "weak"),
("A food delivery app that randomly changes your order.", "FoodTech", "weak"),
("A CRM system that only stores customer complaints, not contacts.", "SaaS", "weak"),
("A weather app that only shows weather for cities you've never visited.", "Consumer", "weak"),
("A platform for renting umbrellas by the minute.", "Marketplace", "weak"),
("An AI writing tool that makes your emails exactly 17% longer.", "AI/SaaS", "weak"),
("A fitness tracker that only counts calories burned while sleeping.", "HealthTech", "weak"),
("A social app where your profile picture must be a vegetable.", "Social", "weak"),
("A travel booking site exclusively for trips under 5 miles.", "Travel", "weak"),
("A music streaming service that only plays songs in reverse.", "Entertainment", "weak"),
("A project management tool where all tasks are assigned randomly.", "SaaS", "weak"),
("A dating app that matches people based on their blood type.", "Social", "weak"),
("An e-commerce store selling artisanal dirt from famous locations.", "E-commerce", "weak"),
("A video conferencing tool where everyone appears as a potato.", "SaaS", "weak"),
("A smart water bottle that insults you when you don't drink enough.", "Hardware", "weak"),
("An app that converts your text messages into Morse code before sending.", "Consumer", "weak"),
("A subscription service for monthly deliveries of bubble wrap.", "E-commerce", "weak"),
("A platform connecting people who want to watch paint dry together.", "Social", "weak"),
("An AI that generates excuses for being late to meetings.", "AI/Consumer", "weak"),
("A marketplace for trading homemade soap shaped like celebrities.", "E-commerce", "weak"),
("A browser extension that replaces all images with pictures of Nicolas Cage.", "Consumer", "weak"),
("A SaaS tool that schedules meetings at the worst possible times.", "SaaS", "weak"),
("A wearable that tracks how many doors you open per day.", "Hardware", "weak"),
("A food app that only recommends restaurants you've already been to.", "FoodTech", "weak"),
("A cloud storage service with a maximum capacity of 1 MB.", "SaaS", "weak"),
("An app that sends you a notification every time a celebrity tweets.", "Consumer", "weak"),
("A platform for competitive rock-paper-scissors leagues.", "Gaming", "weak"),
("A smart pen that only writes in invisible ink.", "Hardware", "weak"),
("A subscription box of random keys that don't open anything.", "E-commerce", "weak"),
("A task management app where completed tasks reappear the next day.", "SaaS", "weak"),
("An AI chatbot that responds to every question with another question.", "AI", "weak"),
("A ride-sharing app exclusively for trips to the grocery store.", "Mobility", "weak"),
("A social platform where posts automatically delete after 3 seconds.", "Social", "weak"),
("A fintech app that converts all your savings into pennies.", "Fintech", "weak"),
("A smart home device that randomly turns lights on and off.", "Hardware", "weak"),
("A language app teaching you to speak like a pirate.", "EdTech", "weak"),
("A job board for positions that require no skills and pay nothing.", "HR", "weak"),
("A meal kit service where all ingredients are dehydrated.", "FoodTech", "weak"),
("A VR game where you simulate filing taxes.", "Gaming", "weak"),
("A platform for rating and reviewing public restrooms by decor only.", "Consumer", "weak"),
("An AI tool that predicts the weather by analyzing your mood.", "AI", "weak"),
("A marketplace for trading expired coupons.", "E-commerce", "weak"),
("A fitness app that only works when you're standing perfectly still.", "HealthTech", "weak"),
("A smart fridge that orders food you're allergic to.", "Hardware", "weak"),
("A social network exclusively for people named Dave.", "Social", "weak"),
("A SaaS tool that converts spreadsheets into interpretive dance instructions.", "SaaS", "weak"),
("A travel app that only suggests destinations during their worst weather season.", "Travel", "weak"),
("A podcast network exclusively for ASMR recordings of office equipment.", "Entertainment", "weak"),
("An app that measures the exact curvature of bananas using AR.", "Consumer", "weak"),
("A B2B platform for outsourcing high-fives at corporate events.", "Services", "weak"),
("A wearable ring that glows when Mercury is in retrograde.", "Hardware", "weak"),
("A subscription service delivering a single puzzle piece per month.", "E-commerce", "weak"),
("An AI that writes apology letters to your houseplants for not watering them.", "AI/Consumer", "weak"),
("A dating app that only matches people who have the same shoe size.", "Social", "weak"),
("A cloud-based platform for storing and sharing your dreams.", "SaaS", "weak"),
("A smart toothbrush that posts your brushing stats to social media.", "Hardware", "weak"),
("A marketplace for trading hand-drawn maps of fictional places.", "E-commerce", "weak"),
("An app that translates baby cries into stock market predictions.", "AI/Fintech", "weak"),
("A SaaS dashboard that visualizes how many unread emails you have as abstract art.", "SaaS", "weak"),
("A delivery service for sending glitter bombs to your enemies.", "E-commerce", "weak"),
("A fitness app that only counts steps taken backwards.", "HealthTech", "weak"),
("A browser extension that replaces all numbers on websites with Roman numerals.", "Consumer", "weak"),
("A platform for competitive speed-reading of terms and conditions.", "Entertainment", "weak"),
("An AI tool that generates corporate jargon for any simple sentence.", "AI/SaaS", "weak"),
("A smart umbrella that closes automatically when it starts raining.", "Hardware", "weak"),
("A social app where you can only communicate using emojis from 2015.", "Social", "weak"),
("A meal planning app that only suggests foods starting with the letter Q.", "FoodTech", "weak"),
("A VR experience of being a traffic cone.", "Entertainment", "weak"),
("A SaaS tool that automatically adds 'per my last email' to all your replies.", "SaaS", "weak"),
("A marketplace for renting out your driveway to people who just want to sit in it.", "Marketplace", "weak"),
("An app that rates sunsets on a 100-point scale using AI.", "AI/Consumer", "weak"),
("A subscription box of motivational quotes printed on sandpaper.", "E-commerce", "weak"),
("A platform connecting people who want to argue about pineapple on pizza.", "Social", "weak"),
("A smart watch that only tells time in a random timezone.", "Hardware", "weak"),
("A fintech app that charges you $1 every time you check your balance.", "Fintech", "weak"),
("An AI writing assistant that makes everything sound like Shakespeare.", "AI", "weak"),
("A job board exclusively for unpaid internships at startups that haven't launched.", "HR", "weak"),
("A food delivery app that delivers meals from 3 days ago at a discount.", "FoodTech", "weak"),
("A platform for reviewing and rating different types of silence.", "Consumer", "weak"),
("A smart doormat that greets visitors with a different insult each time.", "Hardware", "weak"),

# --- BELOW AVERAGE (overall ~3.5-5.0) ---
("A subscription box of curated artisanal hot sauces from around the world.", "E-commerce/DTC", "below_avg"),
("An online marketplace for buying and selling used textbooks between college students.", "Marketplace/EdTech", "below_avg"),
("A social network for vintage typewriter collectors to trade and share collections.", "Social/Niche", "below_avg"),
("A dating app exclusively for dog owners where matches are based on dog compatibility.", "Social/Consumer", "below_avg"),
("A wearable device monitoring soil moisture and alerting home gardeners.", "Hardware/IoT", "below_avg"),
("A Chrome extension blocking social media during work hours with a schedule.", "Productivity", "below_avg"),
("A mobile game where players compete to identify bird species from audio.", "Gaming/EdTech", "below_avg"),
("A decentralized protocol for verifying academic credentials on blockchain.", "Web3/EdTech", "below_avg"),
("A service renting portable soundproof phone booths to coworking spaces.", "Hardware/Services", "below_avg"),
("A platform for booking mobile car detailing services on-demand.", "Marketplace/Services", "below_avg"),
("A browser extension summarizing Terms of Service and highlighting concerning clauses.", "Consumer/LegalTech", "below_avg"),
("An app gamifying personal finance education for teenagers with simulated investing.", "Fintech/EdTech", "below_avg"),
("A smart trash can using AI to sort recyclables from waste automatically.", "Hardware/CleanTech", "below_avg"),
("An app matching people with compatible hiking partners based on fitness and pace.", "Social/Fitness", "below_avg"),
("A platform for fractional ownership of rare sneakers as investment assets.", "Fintech/Consumer", "below_avg"),
("A marketplace for local artists to sell custom phone cases.", "E-commerce", "below_avg"),
("An app that helps people find quiet cafes to work from based on real-time noise levels.", "Consumer/Productivity", "below_avg"),
("A subscription service for curated playlists based on your astrological sign.", "Entertainment", "below_avg"),
("A platform connecting amateur photographers with people who need headshots.", "Marketplace", "below_avg"),
("An AI tool that generates personalized workout playlists based on heart rate.", "HealthTech/AI", "below_avg"),
("A marketplace for renting designer handbags for special occasions.", "E-commerce/Fashion", "below_avg"),
("An app that helps roommates split household chores fairly using an algorithm.", "Consumer", "below_avg"),
("A platform for booking local tour guides for off-the-beaten-path experiences.", "Travel/Marketplace", "below_avg"),
("A SaaS tool for managing community garden plots and shared resources.", "SaaS/Community", "below_avg"),
("An app that connects pet owners with local pet-friendly businesses and events.", "Consumer/Marketplace", "below_avg"),
("A platform for indie board game designers to playtest games with remote players.", "Gaming/Marketplace", "below_avg"),
("A subscription service delivering monthly craft beer from local microbreweries.", "E-commerce/DTC", "below_avg"),
("An AI-powered app that suggests outfit combinations from your existing wardrobe.", "AI/Fashion", "below_avg"),
("A platform connecting retired teachers with students for affordable tutoring.", "EdTech/Marketplace", "below_avg"),
("A smart plant pot that automatically waters and adjusts light for indoor plants.", "Hardware/Consumer", "below_avg"),
("An app for tracking and trading loyalty points across different store programs.", "Fintech/Consumer", "below_avg"),
("A marketplace for commissioning custom illustrations and artwork.", "Creative/Marketplace", "below_avg"),
("A platform for neighborhood tool sharing and lending.", "Marketplace/Community", "below_avg"),
("An app that helps people find and join local sports pickup games.", "Social/Fitness", "below_avg"),
("A SaaS tool for small churches to manage donations, events, and member communication.", "SaaS/Vertical", "below_avg"),
("A platform for connecting home cooks with neighbors who want home-cooked meals.", "FoodTech/Marketplace", "below_avg"),
("An app that tracks your caffeine intake and suggests optimal coffee timing.", "HealthTech/Consumer", "below_avg"),
("A marketplace for vintage and retro video game cartridges.", "E-commerce/Gaming", "below_avg"),
("A platform for booking and reviewing mobile massage therapists.", "Marketplace/Wellness", "below_avg"),
("An AI tool that generates bedtime stories personalized with your child's name.", "AI/Consumer", "below_avg"),
("A SaaS tool for managing shared vacation home schedules among family members.", "SaaS/Consumer", "below_avg"),
("A platform connecting people who want to practice foreign languages over coffee.", "EdTech/Social", "below_avg"),
("An app for tracking and cataloging your personal book collection.", "Consumer", "below_avg"),
("A marketplace for local farmers to sell directly to consumers via weekly boxes.", "FoodTech/Marketplace", "below_avg"),
("A platform for connecting dog walkers with dog owners in the same neighborhood.", "Marketplace/Services", "below_avg"),
("An app that helps people find free parking spots using crowdsourced data.", "Consumer/Mobility", "below_avg"),
("A SaaS tool for yoga studios to manage class schedules, bookings, and payments.", "SaaS/Vertical", "below_avg"),
("A platform for renting camera equipment from local photographers.", "Marketplace", "below_avg"),
("An AI-powered journaling app that provides weekly mental health insights.", "HealthTech/AI", "below_avg"),
("A marketplace for handmade pet accessories and custom pet portraits.", "E-commerce/Pets", "below_avg"),

# --- AVERAGE (overall ~5.0-6.5) ---
("A SaaS tool that auto-generates API documentation from source code.", "SaaS/DevTools", "average"),
("A platform aggregating and comparing insurance quotes for gig workers.", "InsurTech", "average"),
("A telehealth platform for mental health therapy in rural areas.", "HealthTech", "average"),
("A platform connecting retired professionals with startups for advisory roles.", "Marketplace/HR", "average"),
("An app using computer vision to scan receipts and track nutritional intake.", "HealthTech/AI", "average"),
("A drone-based service for inspecting solar panel installations.", "Hardware/CleanTech", "average"),
("A platform connecting homeowners with vetted contractors for small repairs.", "Marketplace/HomeServices", "average"),
("An AI resume screening tool that removes bias indicators before review.", "AI/HR", "average"),
("A vertical SaaS for managing veterinary clinics.", "SaaS/Vertical", "average"),
("A platform for independent musicians to license music to content creators.", "Marketplace/Music", "average"),
("An enterprise tool monitoring employee burnout via communication patterns.", "SaaS/HR", "average"),
("A SaaS platform for managing influencer marketing campaigns end-to-end.", "SaaS/Marketing", "average"),
("A subscription delivering pre-portioned meal kits for specific medical diets.", "FoodTech/HealthTech", "average"),
("An AI writing assistant for academic research papers in STEM fields.", "AI/EdTech", "average"),
("A no-code platform for creating interactive product demos for SaaS companies.", "SaaS/DevTools", "average"),
("A platform for small businesses to manage and respond to online reviews with AI.", "SaaS/SMB", "average"),
("An API service converting any PDF document into structured queryable data.", "SaaS/DevTools", "average"),
("A marketplace for fractional ownership of commercial real estate.", "Fintech/PropTech", "average"),
("A peer-to-peer lending platform for small businesses in emerging markets.", "Fintech", "average"),
("An AI copilot for real estate agents automating listings and lead follow-up.", "PropTech/AI", "average"),
("A B2B platform connecting restaurants with local farms for direct sourcing.", "Marketplace/FoodTech", "average"),
("A SaaS tool detecting and removing duplicate customer records across CRMs.", "SaaS/DataOps", "average"),
("A platform for corporate teams to book and manage offsite retreats.", "SaaS/HR", "average"),
("An AI-powered legal document review tool for small law firms.", "LegalTech/AI", "average"),
("A SaaS platform for property managers to handle maintenance requests and tenant communication.", "PropTech/SaaS", "average"),
("A marketplace connecting freelance CFOs with startups needing part-time finance leadership.", "Marketplace/Fintech", "average"),
("An AI tool that monitors brand mentions across social media and generates sentiment reports.", "SaaS/Marketing", "average"),
("A platform for small e-commerce brands to manage returns and exchanges.", "SaaS/E-commerce", "average"),
("A SaaS tool for restaurants to manage inventory, reduce waste, and auto-reorder supplies.", "SaaS/FoodTech", "average"),
("An AI-powered customer support chatbot builder for small businesses.", "AI/SaaS", "average"),
("A platform connecting certified translators with businesses for document translation.", "Marketplace/Services", "average"),
("A SaaS tool for managing employee onboarding workflows and compliance training.", "SaaS/HR", "average"),
("An AI tool that generates product descriptions for e-commerce listings from photos.", "AI/E-commerce", "average"),
("A platform for booking and managing corporate catering from local restaurants.", "Marketplace/FoodTech", "average"),
("A SaaS tool for nonprofits to manage donor relationships, campaigns, and grants.", "SaaS/Nonprofit", "average"),
("An AI-powered code review tool that catches security vulnerabilities.", "DevTools/AI", "average"),
("A platform connecting physical therapists with patients for remote rehab sessions.", "HealthTech", "average"),
("A SaaS tool for managing construction project timelines, budgets, and subcontractors.", "SaaS/Construction", "average"),
("An AI tool that generates social media content calendars from a brand's past posts.", "AI/Marketing", "average"),
("A platform for small manufacturers to find and compare raw material suppliers.", "Marketplace/B2B", "average"),
("A SaaS tool for managing co-living spaces including billing, maintenance, and community.", "PropTech/SaaS", "average"),
("An AI-powered tool that transcribes and summarizes sales calls with action items.", "AI/SaaS", "average"),
("A platform connecting independent pharmacies with wholesale drug distributors.", "HealthTech/Marketplace", "average"),
("A SaaS tool for event planners to manage vendors, timelines, and budgets.", "SaaS/Events", "average"),
("An AI tool that generates personalized email sequences for sales outreach.", "AI/SaaS", "average"),
("A platform for small businesses to pool purchasing power for better supplier rates.", "Marketplace/B2B", "average"),
("A SaaS tool for managing franchise operations across multiple locations.", "SaaS/Franchise", "average"),
("An AI-powered tool that detects and prevents ad fraud for digital advertisers.", "AI/AdTech", "average"),
("A platform connecting independent insurance agents with carriers for faster quoting.", "InsurTech/Marketplace", "average"),
("A SaaS tool for managing clinical trials including patient recruitment and data collection.", "HealthTech/SaaS", "average"),

# --- STRONG (overall ~6.5-8.0) ---
("An AI-powered platform that automatically generates and files patent applications.", "LegalTech/AI", "strong"),
("A SaaS platform for automating accounts payable and invoice processing for mid-market companies.", "Fintech/SaaS", "strong"),
("An AI-powered cybersecurity platform that detects and responds to threats in real-time for SMBs.", "Cybersecurity/AI", "strong"),
("A vertical SaaS for dental practices managing appointments, billing, imaging, and patient records.", "HealthTech/SaaS", "strong"),
("A platform automating compliance monitoring and reporting for financial institutions.", "RegTech/Fintech", "strong"),
("An AI-powered supply chain optimization platform for mid-market manufacturers.", "AI/Supply Chain", "strong"),
("A developer platform providing pre-built authentication, payments, and notifications as APIs.", "DevTools/SaaS", "strong"),
("A SaaS platform for managing commercial fleet vehicles including maintenance, routing, and compliance.", "SaaS/Logistics", "strong"),
("An AI-powered medical imaging analysis tool that assists radiologists in detecting anomalies.", "HealthTech/AI", "strong"),
("A platform automating tax preparation and filing for small businesses across multiple states.", "Fintech/SaaS", "strong"),
("An AI-powered recruiting platform that matches candidates to roles based on skills, not resumes.", "AI/HR", "strong"),
("A SaaS platform for managing subscription billing, revenue recognition, and churn analytics.", "Fintech/SaaS", "strong"),
("An AI tool that monitors and optimizes cloud infrastructure costs across AWS, GCP, and Azure.", "DevTools/AI", "strong"),
("A platform providing embedded lending and BNPL solutions for B2B e-commerce platforms.", "Fintech", "strong"),
("A SaaS tool for managing multi-location retail operations including inventory, staff, and POS.", "SaaS/Retail", "strong"),
("An AI-powered contract analysis tool that extracts key terms and flags risks for legal teams.", "LegalTech/AI", "strong"),
("A platform for automating insurance claims processing using AI and document analysis.", "InsurTech/AI", "strong"),
("A SaaS tool for managing clinical workflows and electronic health records for specialty clinics.", "HealthTech/SaaS", "strong"),
("An AI-powered fraud detection platform for online payments and e-commerce transactions.", "Fintech/AI", "strong"),
("A developer tool that automatically generates and maintains database migration scripts.", "DevTools", "strong"),
("A SaaS platform for managing warehouse operations including picking, packing, and shipping.", "SaaS/Logistics", "strong"),
("An AI tool that personalizes e-commerce product recommendations based on browsing behavior.", "AI/E-commerce", "strong"),
("A platform providing real-time language translation for customer support across channels.", "AI/SaaS", "strong"),
("A SaaS tool for managing and optimizing digital advertising spend across multiple platforms.", "SaaS/AdTech", "strong"),
("An AI-powered predictive maintenance platform for industrial equipment.", "AI/Industrial", "strong"),
("A platform automating accounts receivable and collections for B2B companies.", "Fintech/SaaS", "strong"),
("A SaaS tool for managing food safety compliance and inspections for restaurant chains.", "SaaS/FoodTech", "strong"),
("An AI tool that generates and optimizes landing pages based on conversion data.", "AI/Marketing", "strong"),
("A platform providing embedded insurance products for e-commerce and gig platforms.", "InsurTech", "strong"),
("A SaaS tool for managing and automating procurement workflows for enterprises.", "SaaS/Procurement", "strong"),
("A developer platform providing serverless database and backend infrastructure.", "DevTools/Infrastructure", "strong"),
("An AI-powered tool that monitors regulatory changes and alerts affected businesses.", "RegTech/AI", "strong"),
("A SaaS platform for managing loyalty programs and customer rewards across retail chains.", "SaaS/Retail", "strong"),
("An AI tool that automates financial modeling and scenario analysis for CFOs.", "AI/Fintech", "strong"),
("A platform providing identity verification and KYC services via API for fintech companies.", "Fintech/Identity", "strong"),
("A SaaS tool for managing and optimizing last-mile delivery operations.", "SaaS/Logistics", "strong"),
("An AI-powered tool that generates video content from text scripts for marketing teams.", "AI/Marketing", "strong"),
("A platform automating payroll processing and tax compliance for companies with global teams.", "Fintech/HR", "strong"),
("A SaaS tool for managing energy consumption and sustainability reporting for commercial buildings.", "SaaS/CleanTech", "strong"),
("An AI tool that detects and prevents account takeover fraud for online platforms.", "Cybersecurity/AI", "strong"),
("A platform providing embedded fintech infrastructure (payments, lending, cards) for SaaS companies.", "Fintech/Infrastructure", "strong"),
("A SaaS tool for managing clinical documentation and coding for healthcare providers.", "HealthTech/SaaS", "strong"),
("An AI-powered tool that optimizes pricing dynamically based on demand, competition, and inventory.", "AI/E-commerce", "strong"),
("A platform automating vendor risk assessment and third-party compliance monitoring.", "SaaS/Compliance", "strong"),
("A developer tool providing real-time error monitoring, alerting, and debugging for production apps.", "DevTools", "strong"),
("A SaaS platform for managing and automating employee benefits administration.", "SaaS/HR", "strong"),
("An AI tool that generates architectural floor plans from natural language descriptions.", "AI/PropTech", "strong"),
("A platform providing data integration and ETL pipelines as a managed service.", "DevTools/Data", "strong"),
("A SaaS tool for managing and optimizing field service operations for utility companies.", "SaaS/Utilities", "strong"),
("An AI-powered tool that automates due diligence document review for M&A transactions.", "AI/Fintech", "strong"),
("A platform providing real-time freight rate comparison and booking for shippers.", "Marketplace/Logistics", "strong"),

# --- EXCEPTIONAL (overall ~8.0-9.5) ---
("An AI-powered platform that automates end-to-end drug discovery from target identification to lead optimization.", "HealthTech/AI", "exceptional"),
("A developer infrastructure platform providing globally distributed edge computing with sub-10ms latency.", "DevTools/Infrastructure", "exceptional"),
("An AI-powered cybersecurity platform that autonomously detects, investigates, and remediates threats across enterprise networks.", "Cybersecurity/AI", "exceptional"),
("A fintech platform providing real-time cross-border B2B payments with automatic FX hedging and compliance.", "Fintech", "exceptional"),
("An AI platform that enables any company to build and deploy custom foundation models on their proprietary data.", "AI/Infrastructure", "exceptional"),
("A vertical SaaS platform that digitizes and automates the entire commercial insurance underwriting workflow.", "InsurTech/SaaS", "exceptional"),
("An AI-powered platform that generates production-ready mobile apps from natural language descriptions.", "AI/DevTools", "exceptional"),
("A platform providing embedded banking infrastructure (accounts, cards, lending) for any SaaS company via API.", "Fintech/Infrastructure", "exceptional"),
("An AI-powered drug repurposing platform that identifies new therapeutic uses for existing approved drugs.", "HealthTech/AI", "exceptional"),
("A developer platform providing automated security scanning, compliance, and remediation for cloud infrastructure.", "DevTools/Security", "exceptional"),
("An AI platform that automates the entire financial audit process for public companies.", "AI/Fintech", "exceptional"),
("A SaaS platform that provides real-time supply chain visibility and predictive disruption alerts across global networks.", "SaaS/Supply Chain", "exceptional"),
("An AI-powered platform for autonomous vehicle fleet management including routing, maintenance, and safety monitoring.", "AI/Mobility", "exceptional"),
("A platform providing programmable telecom infrastructure (voice, SMS, video) via API for developers.", "DevTools/Telecom", "exceptional"),
("An AI-powered clinical decision support system that assists doctors with diagnosis and treatment recommendations.", "HealthTech/AI", "exceptional"),
("A fintech platform automating treasury management including cash forecasting, investment, and risk management.", "Fintech/SaaS", "exceptional"),
("An AI platform that generates synthetic training data for machine learning models across any domain.", "AI/Infrastructure", "exceptional"),
("A SaaS platform providing end-to-end carbon accounting, reduction planning, and offset management for enterprises.", "SaaS/CleanTech", "exceptional"),
("An AI-powered platform that automates the entire mortgage origination process from application to closing.", "Fintech/AI", "exceptional"),
("A developer platform providing unified observability (logs, metrics, traces) with AI-powered root cause analysis.", "DevTools/Infrastructure", "exceptional"),
("An AI platform that enables real-time language translation for live video calls with lip-sync.", "AI/Communication", "exceptional"),
("A SaaS platform automating the entire commercial real estate transaction lifecycle.", "PropTech/SaaS", "exceptional"),
("An AI-powered platform for personalized cancer treatment planning based on genomic analysis.", "HealthTech/AI", "exceptional"),
("A fintech platform providing instant business credit decisions using real-time accounting and banking data.", "Fintech/AI", "exceptional"),
("A developer platform providing AI-powered database optimization that automatically tunes queries and indexes.", "DevTools/AI", "exceptional"),
("An AI platform that automates regulatory compliance reporting across multiple jurisdictions for global banks.", "RegTech/AI", "exceptional"),
("A SaaS platform for managing and optimizing renewable energy assets including solar, wind, and battery storage.", "SaaS/Energy", "exceptional"),
("An AI-powered platform that detects and prevents money laundering across financial networks in real-time.", "Fintech/AI", "exceptional"),
("A platform providing programmable payment infrastructure for marketplaces including splits, escrow, and payouts.", "Fintech/Infrastructure", "exceptional"),
("An AI-powered platform that automates the entire clinical trial matching process for oncology patients.", "HealthTech/AI", "exceptional"),
]



# ============================================================================
# SCORE GENERATION BY TIER
# ============================================================================

# Reason templates per dimension per tier
REASON_TEMPLATES = {
    "pain_severity": {
        "weak": ["No real problem being solved", "Novelty, not a pain point", "Creates friction rather than solving it", "Addresses a non-existent need", "Trivial inconvenience at best"],
        "below_avg": ["Mild inconvenience but not critical", "Nice-to-have, not must-have", "Problem exists but is low priority", "Hobby-level need, not urgent", "Some frustration but manageable"],
        "average": ["Moderate pain that affects productivity", "Real problem but not top priority for most", "Meaningful friction in current workflows", "Costs time and money but workarounds exist", "Growing pain as the market matures"],
        "strong": ["Significant pain that directly impacts revenue or operations", "Critical workflow bottleneck for target users", "Costly problem that compounds over time", "Regulatory or compliance risk creates urgency", "Direct impact on business outcomes"],
        "exceptional": ["Mission-critical problem with severe consequences if unsolved", "Existential risk for businesses without a solution", "Multi-billion dollar problem affecting entire industries", "Life-or-death implications in healthcare or safety", "Regulatory mandate creates forced adoption"],
    },
    "pain_frequency": {
        "weak": ["Rarely or never encountered", "One-time novelty", "No recurring need", "Occasional at best", "Seasonal or event-based only"],
        "below_avg": ["A few times per year", "Occasional but not regular", "Seasonal need", "Triggered by specific events", "Monthly at most"],
        "average": ["Weekly occurrence for target users", "Regular part of business operations", "Multiple times per month", "Ongoing but not daily", "Cyclical with business rhythms"],
        "strong": ["Daily workflow requirement", "Multiple times per day for power users", "Continuous monitoring needed", "Every transaction triggers this need", "Constant operational requirement"],
        "exceptional": ["Real-time, always-on requirement", "Every second matters in this domain", "Continuous 24/7 monitoring essential", "Every interaction requires this", "Mission-critical uptime requirement"],
    },
    "existing_alternatives": {
        "weak": ["Dozens of identical solutions exist", "Completely saturated market", "Free alternatives are excellent", "Built-in OS/browser features cover this", "No differentiation possible"],
        "below_avg": ["Several good alternatives exist", "Established players serve this well", "Free tools cover 80% of the need", "Low switching costs from current solutions", "Crowded market with clear leaders"],
        "average": ["Alternatives exist but have significant gaps", "Current solutions are expensive or complex", "Market is served but not well", "Incumbents are slow to innovate", "Partial solutions exist but nothing end-to-end"],
        "strong": ["Few direct competitors and they have major limitations", "Existing solutions are outdated or overpriced", "No purpose-built solution for this specific segment", "Current approaches are manual and error-prone", "Incumbents focused on enterprise, leaving SMB underserved"],
        "exceptional": ["No viable solution exists today", "Current approaches are fundamentally broken", "Greenfield market with no direct competitors", "Existing tools cannot handle the scale or complexity required", "Regulatory changes have created a new category"],
    },
    "willingness_to_pay": {
        "weak": ["Users expect this to be free", "No perceived value worth paying for", "Free alternatives are good enough", "Target audience has no budget", "Novelty doesn't justify payment"],
        "below_avg": ["Some might pay a small amount", "Price-sensitive audience", "Freemium model might work but conversion would be low", "Willingness to pay is uncertain", "Low perceived value relative to alternatives"],
        "average": ["Moderate willingness to pay for clear value", "B2B buyers have budgets for this category", "Subscription model viable at $20-100/month", "Clear ROI justifies the cost", "Competitive pricing pressure limits premium"],
        "strong": ["Strong willingness to pay for proven ROI", "Enterprise budgets allocated for this category", "High-value problem justifies premium pricing", "Customers currently paying more for inferior solutions", "Clear cost savings or revenue increase justifies price"],
        "exceptional": ["Customers eagerly pay premium for best-in-class", "Mission-critical spending with large budgets", "Regulatory compliance makes this a must-buy", "ROI is 10x+ the cost of the solution", "Customers would pay significantly more than current pricing"],
    },
    "market_size": {
        "weak": ["No meaningful addressable market", "Hyper-niche with fewer than 1000 potential users", "Market is shrinking", "Too narrow to sustain a business", "Addressable segment is negligible"],
        "below_avg": ["Small niche market", "Limited to a specific geography or demographic", "Market exists but is small", "Growing slowly", "Addressable market under $100M"],
        "average": ["Meaningful market with room for multiple players", "Growing market driven by secular trends", "Addressable market in the $100M-$1B range", "Large potential but requires market education", "Expanding as the industry digitizes"],
        "strong": ["Large addressable market with strong growth", "Multi-billion dollar TAM", "Global market with cross-border potential", "Expanding rapidly due to regulatory or technology shifts", "Multiple customer segments to expand into"],
        "exceptional": ["Massive global market measured in tens of billions", "Every company in the target segment is a potential customer", "Market is growing 30%+ annually", "Platform opportunity with network effects driving expansion", "Foundational infrastructure that every business needs"],
    },
    "scalability": {
        "weak": ["Physical product or service that doesn't scale", "Requires linear headcount growth", "Geographic constraints limit expansion", "High marginal cost per customer", "Cannot serve more customers without proportional cost increase"],
        "below_avg": ["Some scalability but significant operational overhead", "Marketplace requires building both sides in each market", "Hardware component limits pure software scaling", "Service delivery requires trained professionals", "Moderate marginal costs"],
        "average": ["SaaS model with reasonable unit economics at scale", "Platform scales but requires ongoing content or data investment", "Moderate infrastructure costs that grow sub-linearly", "Can expand to new segments with some customization", "Scalable core with some manual processes"],
        "strong": ["Pure software with near-zero marginal cost", "API-based model scales horizontally", "Self-serve onboarding reduces sales costs", "Network effects improve the product as it scales", "Cloud-native architecture handles growth efficiently"],
        "exceptional": ["Platform with strong network effects and viral growth", "Infrastructure play that becomes more valuable with scale", "Zero marginal cost with usage-based pricing", "Flywheel effect where more data improves the product", "Winner-take-most dynamics in the category"],
    },
    "profitability_potential": {
        "weak": ["No viable monetization path", "Negative unit economics", "Free market with no willingness to pay", "Costs exceed any possible revenue", "Commodity product with race-to-bottom pricing"],
        "below_avg": ["Thin margins on physical goods", "High customer acquisition costs relative to LTV", "Competitive pricing pressure limits margins", "Revenue possible but profitability uncertain", "Low average contract value"],
        "average": ["Viable SaaS margins of 60-70%", "Clear path to profitability with scale", "Multiple revenue streams possible", "Moderate customer acquisition costs", "Healthy unit economics once past initial investment"],
        "strong": ["High-margin SaaS with 75%+ gross margins", "Strong unit economics with LTV/CAC > 3x", "Enterprise pricing with large contract values", "Expansion revenue from existing customers", "Platform fees on high-volume transactions"],
        "exceptional": ["90%+ gross margins at scale", "Network effects reduce CAC over time", "Pricing power from mission-critical positioning", "Multiple monetization vectors (subscription + usage + marketplace)", "Winner-take-most economics with increasing returns"],
    },
    "defensibility": {
        "weak": ["Trivially copyable with no barriers", "No proprietary technology or data", "Feature, not a product", "Any competitor can replicate in weeks", "No switching costs for users"],
        "below_avg": ["Some brand loyalty but low switching costs", "First-mover advantage is temporary", "Open-source alternatives could emerge", "Moderate technical complexity but replicable", "Community provides some moat"],
        "average": ["Integration depth creates switching costs", "Proprietary data or algorithms provide advantage", "Regulatory expertise creates barriers", "Network of partnerships is hard to replicate", "Domain expertise embedded in the product"],
        "strong": ["Strong network effects that grow with usage", "Proprietary data moat that compounds over time", "Deep integrations make switching costly", "Regulatory approvals create barriers to entry", "Platform ecosystem with third-party developers"],
        "exceptional": ["Winner-take-most network effects", "Proprietary data advantage that is impossible to replicate", "Multi-year regulatory approval process for competitors", "Deep platform lock-in with ecosystem dependencies", "Foundational infrastructure that becomes industry standard"],
    },
    "time_to_value": {
        "weak": ["No value delivered", "Months before any benefit", "Requires fundamental behavior change", "Complex onboarding with uncertain payoff", "Value proposition is unclear"],
        "below_avg": ["Weeks to see initial value", "Requires significant setup and configuration", "Learning curve before productivity gains", "Need to build critical mass before useful", "Gradual value accumulation"],
        "average": ["Days to initial value with guided onboarding", "Quick setup with templates and defaults", "Value visible within first week of use", "Self-serve trial demonstrates value quickly", "Immediate partial value with full value over time"],
        "strong": ["Hours to first value with streamlined onboarding", "Instant demo shows clear before/after", "Plug-and-play integration with existing tools", "Free trial converts quickly due to obvious value", "Immediate time or cost savings on first use"],
        "exceptional": ["Minutes to first value with zero configuration", "Instant ROI visible on first interaction", "Self-serve signup to value in under an hour", "Viral adoption because value is immediately obvious", "Replaces manual process with instant automation"],
    },
    "founder_market_fit_requirement": {
        "weak": ["No specialized knowledge needed", "Anyone can build this", "Basic programming skills sufficient", "No domain expertise required", "General consumer product"],
        "below_avg": ["Some industry familiarity helpful", "Basic understanding of target market needed", "General business knowledge sufficient", "Light technical skills required", "Community management experience helpful"],
        "average": ["Meaningful domain expertise needed", "Industry connections accelerate growth", "Technical depth in specific area required", "Understanding of regulatory landscape important", "Sales experience in the vertical is valuable"],
        "strong": ["Deep domain expertise required for credibility", "Industry relationships are critical for distribution", "Specialized technical skills in AI/ML or specific domain", "Regulatory knowledge is essential", "Years of experience in the target industry needed"],
        "exceptional": ["World-class expertise in a specialized field required", "PhD-level technical depth needed", "Regulatory and compliance expertise across jurisdictions", "Deep relationships with key industry stakeholders", "Combination of rare technical and domain skills"],
    },
}

SCORE_RANGES = {
    "weak": (1, 3),
    "below_avg": (3, 5),
    "average": (5, 7),
    "strong": (7, 9),
    "exceptional": (8, 10),
}

def generate_scores(tier):
    """Generate randomized but coherent scores for a given tier."""
    lo, hi = SCORE_RANGES[tier]
    sr = {}
    for dim in DIMS:
        # Add some variance: occasionally a dimension scores outside the tier range
        base = random.randint(lo, hi)
        # 20% chance of ±1-2 variance for realism
        if random.random() < 0.2:
            base = max(1, min(10, base + random.choice([-2, -1, 1, 2])))
        # Scalability and time_to_value can be high even for weak ideas (apps scale trivially)
        if dim in ("scalability", "time_to_value") and tier == "weak":
            base = random.randint(1, 9)
        # founder_market_fit is inverse for exceptional (high = hard = needs expertise)
        if dim == "founder_market_fit_requirement" and tier == "exceptional":
            base = random.randint(7, 10)
        reason = random.choice(REASON_TEMPLATES[dim][tier])
        sr[dim] = (base, reason)
    return sr

SUMMARY_TEMPLATES = {
    "weak": [
        "No viable business here. Solves no real problem and has no path to monetization.",
        "Novelty concept with zero commercial potential. Fun as a side project but not a business.",
        "Solution looking for a problem. The target market either doesn't exist or doesn't care.",
        "Fundamentally flawed concept. No pain point, no willingness to pay, no defensibility.",
        "Would not recommend pursuing. The market signals are all negative.",
        "Interesting thought experiment but fails on every business viability dimension.",
        "The core premise doesn't hold up to scrutiny. No evidence of real demand.",
        "Too niche, too trivial, or too absurd to build a sustainable business around.",
    ],
    "below_avg": [
        "Has a kernel of a real idea but execution challenges and market limitations are significant.",
        "Could work as a small lifestyle business but unlikely to achieve venture scale.",
        "Real but small market with established alternatives. Differentiation is the key challenge.",
        "Interesting concept but the economics don't work at scale. Thin margins and high competition.",
        "Addresses a real need but the willingness to pay is uncertain and alternatives exist.",
        "Might find a small audience but growth potential is limited by market size.",
        "The idea has merit but timing, competition, or market dynamics work against it.",
        "Could succeed in a very specific niche but broader expansion would be difficult.",
    ],
    "average": [
        "Solid idea with clear market need. Execution and differentiation will determine success.",
        "Viable business with moderate growth potential. Competition is the main risk.",
        "Good product-market fit potential but needs strong execution to stand out.",
        "Real problem with willing buyers. The challenge is building a defensible position.",
        "Promising opportunity in a growing market. Multiple paths to monetization exist.",
        "Strong fundamentals but faces competition from both incumbents and startups.",
        "The market is there and the pain is real. Success depends on go-to-market strategy.",
        "Decent opportunity with clear unit economics. Scale depends on sales efficiency.",
    ],
    "strong": [
        "Strong opportunity with clear pain point, large market, and viable business model.",
        "High-potential idea with strong unit economics and multiple growth vectors.",
        "Addresses a critical need with high willingness to pay. Defensibility through data and integrations.",
        "Excellent product-market fit indicators. The market is large and growing rapidly.",
        "Strong business fundamentals with clear path to profitability and scale.",
        "Compelling value proposition with measurable ROI for customers. Enterprise-ready.",
        "Well-positioned in a growing market with strong tailwinds. Execution risk is manageable.",
        "High-conviction opportunity. The combination of pain, market size, and timing is favorable.",
    ],
    "exceptional": [
        "Exceptional opportunity at the intersection of massive market need and technological capability.",
        "Category-defining potential. Addresses a multi-billion dollar problem with a fundamentally better approach.",
        "Rare combination of enormous market, critical pain, strong defensibility, and excellent unit economics.",
        "Platform opportunity with winner-take-most dynamics. Early mover advantage is significant.",
        "Transformative potential in a market ready for disruption. Strong technical moat and regulatory barriers.",
        "Best-in-class opportunity. Every dimension scores highly and the timing is right.",
        "Generational opportunity to build critical infrastructure for an entire industry.",
        "Exceptional across all dimensions. The question is execution speed, not market viability.",
    ],
}

# Generate examples from PROJECTS templates
for i, (desc, industry, tier) in enumerate(PROJECTS):
    sr = generate_scores(tier)
    summary = random.choice(SUMMARY_TEMPLATES[tier])
    ALL_EXAMPLES.append(make_example(f"g{i+1:03d}", desc, sr, summary))



# ============================================================================
# PARAPHRASE VARIANTS — same idea, different wording (for consistency training)
# ============================================================================

PARAPHRASE_PAIRS = [
    ("An app that helps freelancers track unpaid invoices and auto-send payment reminders via email and SMS.",
     "Invoice management software for independent contractors that automatically follows up on overdue payments through multiple channels."),
    ("A B2B platform connecting small restaurants with local farms for direct produce sourcing.",
     "A farm-to-table marketplace that lets independent restaurants buy fresh ingredients directly from nearby agricultural producers."),
    ("An AI-powered platform that automatically generates and files patent applications.",
     "Automated patent filing software using artificial intelligence to handle prior art search, claim drafting, and USPTO submission."),
    ("A browser extension that changes all website backgrounds to the color blue.",
     "A simple browser plugin that applies a blue background color to every webpage you visit."),
    ("A SaaS tool that automatically detects and removes duplicate customer records across CRM systems.",
     "Customer data deduplication software that identifies and merges duplicate contacts across multiple CRM platforms."),
    ("A telehealth platform specifically for mental health therapy in rural and underserved areas.",
     "Online therapy service focused on connecting licensed therapists with patients in remote communities lacking mental health providers."),
    ("An API service that converts any PDF document into structured, queryable data.",
     "A document intelligence API that extracts and structures information from PDF files into searchable, machine-readable formats."),
    ("A no-code platform for creating interactive product demos and walkthroughs for SaaS companies.",
     "Demo creation software that lets SaaS marketing teams build clickable product tours without writing any code."),
    ("A platform for independent musicians to license their music directly to content creators.",
     "A music licensing marketplace connecting indie artists with YouTubers, podcasters, and brands who need affordable licensed tracks."),
    ("An enterprise tool that monitors employee burnout risk using anonymized communication pattern analysis.",
     "Workforce wellbeing analytics software that detects burnout signals from anonymized email and chat metadata patterns."),
]

# For each pair, generate the same tier scores (strong/average) with slight variance
for i, (orig, para) in enumerate(PARAPHRASE_PAIRS):
    tier = random.choice(["average", "strong"])
    sr1 = generate_scores(tier)
    # Paraphrase gets similar but not identical scores (±1 on some dims)
    sr2 = {}
    for dim in DIMS:
        s, r = sr1[dim]
        delta = random.choice([-1, 0, 0, 0, 1])  # mostly same, occasionally ±1
        sr2[dim] = (max(1, min(10, s + delta)), r)
    summary = random.choice(SUMMARY_TEMPLATES[tier])
    ALL_EXAMPLES.append(make_example(f"p{i+1:03d}a", orig, sr1, summary, "paraphrase_original"))
    ALL_EXAMPLES.append(make_example(f"p{i+1:03d}b", para, sr2, summary, "paraphrase_variant"))


# ============================================================================
# EDGE CASES
# ============================================================================

EDGE_CASES = [
    make_example("e001", "an app for food",
        {"pain_severity":(3,"Vague description — could address real food-related pain but unclear"),"pain_frequency":(5,"People eat daily"),"existing_alternatives":(2,"Thousands of food apps exist"),"willingness_to_pay":(3,"Depends entirely on what the app does"),"market_size":(5,"Food market is huge but description is too vague"),"scalability":(5,"Unknown without more detail"),"profitability_potential":(3,"Cannot assess without knowing the product"),"defensibility":(2,"No clear differentiation described"),"time_to_value":(4,"Unknown"),"founder_market_fit_requirement":(3,"Depends on the specific food problem")},
        "Description is too vague to evaluate meaningfully. Need specifics on what problem is being solved, for whom, and how."),
    make_example("e002", "We're building the Uber for dogs. It's like Airbnb meets Tinder but for pets. We use blockchain, AI, and quantum computing to disrupt the $500B pet industry. Our proprietary algorithm leverages synergistic paradigm shifts to create a best-in-class, world-class, enterprise-grade solution.",
        {"pain_severity":(3,"Buried under buzzwords — pet care has real pain points but this doesn't articulate one"),"pain_frequency":(4,"Pet care is ongoing"),"existing_alternatives":(3,"Rover and Wag serve pet services well"),"willingness_to_pay":(4,"Pet owners spend on services"),"market_size":(5,"Pet industry is large"),"scalability":(4,"Unclear what the actual product is"),"profitability_potential":(3,"Cannot assess without understanding the business model"),"defensibility":(2,"Buzzword soup suggests no real technical moat"),"time_to_value":(3,"Unclear what value is delivered"),"founder_market_fit_requirement":(4,"Pet industry knowledge needed")},
        "Red flags: buzzword overload, unclear value proposition, claims of multiple cutting-edge technologies without specifics. The pet industry is real but this pitch doesn't articulate a clear problem or solution."),
    make_example("e003", "asdfghjkl qwerty zxcvbn",
        {"pain_severity":(1,"Not a coherent project description"),"pain_frequency":(1,"Cannot evaluate"),"existing_alternatives":(1,"Cannot evaluate"),"willingness_to_pay":(1,"Cannot evaluate"),"market_size":(1,"Cannot evaluate"),"scalability":(1,"Cannot evaluate"),"profitability_potential":(1,"Cannot evaluate"),"defensibility":(1,"Cannot evaluate"),"time_to_value":(1,"Cannot evaluate"),"founder_market_fit_requirement":(1,"Cannot evaluate")},
        "Invalid input. This is not a project description. Cannot provide meaningful evaluation."),
    make_example("e004", "A platform that helps people find and hire hitmen for contract killings.",
        {"pain_severity":(1,"Illegal activity — not a legitimate pain point"),"pain_frequency":(1,"Not applicable"),"existing_alternatives":(1,"Not applicable — illegal market"),"willingness_to_pay":(1,"Not applicable"),"market_size":(1,"Not a legitimate market"),"scalability":(1,"Not applicable"),"profitability_potential":(1,"Illegal business has no legitimate profitability"),"defensibility":(1,"Not applicable"),"time_to_value":(1,"Not applicable"),"founder_market_fit_requirement":(1,"Not applicable")},
        "This describes an illegal activity. Cannot and should not be evaluated as a business opportunity."),
    make_example("e005", "What is the meaning of life?",
        {"pain_severity":(1,"This is a philosophical question, not a project description"),"pain_frequency":(1,"Not applicable"),"existing_alternatives":(1,"Not applicable"),"willingness_to_pay":(1,"Not applicable"),"market_size":(1,"Not applicable"),"scalability":(1,"Not applicable"),"profitability_potential":(1,"Not applicable"),"defensibility":(1,"Not applicable"),"time_to_value":(1,"Not applicable"),"founder_market_fit_requirement":(1,"Not applicable")},
        "This is not a project description. Please provide a description of a product, service, or business idea to evaluate."),
    make_example("e006", ".",
        {"pain_severity":(1,"Empty or minimal input — cannot evaluate"),"pain_frequency":(1,"Cannot evaluate"),"existing_alternatives":(1,"Cannot evaluate"),"willingness_to_pay":(1,"Cannot evaluate"),"market_size":(1,"Cannot evaluate"),"scalability":(1,"Cannot evaluate"),"profitability_potential":(1,"Cannot evaluate"),"defensibility":(1,"Cannot evaluate"),"time_to_value":(1,"Cannot evaluate"),"founder_market_fit_requirement":(1,"Cannot evaluate")},
        "Input is too short to evaluate. Please provide a meaningful project description."),
    make_example("e007", "I want to build something with AI that makes money. Not sure what yet but AI is hot right now and I want to get in on it. Maybe something with ChatGPT or image generation. My friend said I should do a startup.",
        {"pain_severity":(2,"No specific problem identified"),"pain_frequency":(2,"Cannot assess without a defined problem"),"existing_alternatives":(2,"AI space is extremely crowded"),"willingness_to_pay":(2,"No clear value proposition to pay for"),"market_size":(3,"AI market is large but this has no focus"),"scalability":(3,"Depends on what is actually built"),"profitability_potential":(2,"No business model articulated"),"defensibility":(1,"No differentiation without a specific product"),"time_to_value":(2,"No product to deliver value"),"founder_market_fit_requirement":(3,"AI skills needed but no domain focus")},
        "Too vague to evaluate. 'Something with AI' is not a business idea. Needs a specific problem, target customer, and solution before evaluation is meaningful."),
    make_example("e008", "Una aplicación que ayuda a los restaurantes pequeños a gestionar sus pedidos y entregas de forma eficiente, reduciendo el desperdicio de alimentos en un 30%.",
        {"pain_severity":(7,"Food waste and order management are real operational pains for small restaurants"),"pain_frequency":(9,"Restaurants manage orders and inventory daily"),"existing_alternatives":(5,"Toast, Square, and others exist but many small restaurants still use manual processes"),"willingness_to_pay":(7,"Clear ROI from waste reduction justifies subscription cost"),"market_size":(7,"Millions of small restaurants globally"),"scalability":(7,"SaaS model scales well across restaurants"),"profitability_potential":(7,"Subscription revenue with good unit economics"),"defensibility":(4,"Integration with POS and suppliers creates some switching costs"),"time_to_value":(6,"Requires setup and integration with existing workflows"),"founder_market_fit_requirement":(6,"Restaurant operations and food supply chain knowledge needed")},
        "Strong idea with clear ROI proposition. The 30% waste reduction claim needs validation but if achievable, the value proposition is compelling. Competition exists but many small restaurants remain underserved by current solutions. (Note: description was in Spanish — model should handle multilingual input.)"),
]

ALL_EXAMPLES.extend(EDGE_CASES)


# ============================================================================
# ASSEMBLE AND SAVE
# ============================================================================

def main():
    # Load existing hand-crafted data
    existing_file = Path(__file__).parent.parent / "data" / "training_data.json"
    with open(existing_file) as f:
        existing = json.load(f)

    # Merge: existing first, then generated (dedup by source_id)
    seen_ids = {ex.get("source_id") for ex in existing}
    merged = list(existing)
    for ex in ALL_EXAMPLES:
        if ex["source_id"] not in seen_ids:
            merged.append(ex)
            seen_ids.add(ex["source_id"])

    # Shuffle
    random.shuffle(merged)

    # Stats
    scores = []
    for ex in merged:
        ev = json.loads(ex["messages"][2]["content"])
        scores.append(ev.get("overall_score", 0))

    print(f"Total examples: {len(merged)}")
    print(f"Score range: {min(scores):.1f} - {max(scores):.1f}")
    print(f"Mean score: {sum(scores)/len(scores):.1f}")

    # Distribution
    buckets = {"1-3": 0, "3.1-5": 0, "5.1-6.5": 0, "6.6-8": 0, "8.1-10": 0}
    for s in scores:
        if s <= 3: buckets["1-3"] += 1
        elif s <= 5: buckets["3.1-5"] += 1
        elif s <= 6.5: buckets["5.1-6.5"] += 1
        elif s <= 8: buckets["6.6-8"] += 1
        else: buckets["8.1-10"] += 1

    print(f"\nScore distribution:")
    for bucket, count in buckets.items():
        pct = 100 * count / len(merged)
        bar = "█" * int(pct / 2)
        print(f"  {bucket:>8s}: {count:>4d} ({pct:5.1f}%) {bar}")

    # Source type distribution
    from collections import Counter
    types = Counter(ex.get("source_type", "unknown") for ex in merged)
    print(f"\nSource types: {dict(types)}")

    # Save
    output = Path(__file__).parent.parent / "data" / "training_data_full.json"
    with open(output, "w") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to: {output}")


if __name__ == "__main__":
    main()
