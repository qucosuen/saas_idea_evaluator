"""Scrape Indie Hackers ideas and upload to Postgres + update feature store."""
import json, re, psycopg2
from psycopg2.extras import execute_values
from datetime import datetime, timezone
from pathlib import Path

# Parsed from https://www.indiehackers.com/ideas (fetched 2026-04-13)
RAW = """GymStreak|AI-powered fitness app using 3D animations and AR for workout guidance|Health & Wellness,Mobile Apps,3D Printing|$208K+ MRR
Chatbase|Platform for businesses to create AI-powered customer support agents|Artificial Intelligence (AI),Customer Service,SaaS|$417K+ MRR
Interview Coder|AI tool that assists candidates in passing coding interviews|Artificial Intelligence (AI),Career Development,Software Development|$110K MRR
Advocat AI|AI-driven tool for managing and optimizing contract workflows|Artificial Intelligence (AI),Legal,Automation|$80K+ MRR
Kore.ai|No-code platform for building conversational AI solutions across industries|Artificial Intelligence (AI),No-Code,SaaS|$8MM+ MRR
Castmagic|AI-driven tool that converts audio content into written formats|Artificial Intelligence (AI),Audio,Content Creation|$120K+ MRR
Photoroom|AI-powered photo editing app for entrepreneurs and small businesses|Artificial Intelligence (AI),Photography,Entrepreneurship|$4MM+ MRR
Uizard|AI-powered design tool that simplifies UI creation for non-designers|Design,Artificial Intelligence (AI),Graphic Design|$290K+ MRR
Salesforge|AI-powered tool that improves B2B cold email deliverability|Email,Artificial Intelligence (AI),Sales|$167K+ MRR
SkillSoniq|AI-powered recruiting app connecting companies with domestic freelancers|Artificial Intelligence (AI),Hiring,Mobile Apps|$40K+ MRR
Enterprise Bot|AI chatbot platform for businesses to enhance customer interactions|Artificial Intelligence (AI),Automation,SaaS|$150K MRR
EasyGen|AI-powered tool for generating LinkedIn content for professionals|Artificial Intelligence (AI),Content Creation,Social Media|$45K MRR
Fireflies.ai|AI-powered app that transcribes and organizes meeting notes|Artificial Intelligence (AI),Productivity,Remote Work|$500K+ MRR
Lovable|AI-powered coding assistant that helps non-developers create software|Artificial Intelligence (AI),Software Development,Productivity|$330K+ MRR
Rezi|AI-powered resume builder that optimizes resumes for job applications|Artificial Intelligence (AI),Career Development,Productivity|$225K+ MRR
Float|Resource scheduling app for agencies to manage team projects efficiently|Project Management,Productivity,SaaS|$230K+ MRR
Chili Piper|Platform that helps sales teams qualify, route, and schedule meetings with leads|Sales,Productivity,SaaS|$2.5MM+ MRR
Wappalyzer|Browser extension that identifies website technologies for developers|Chrome Extension,Developer Tools,Software Development|$85K+ MRR
vFairs|Virtual events platform enabling organizations to host engaging online events|Events & Conferencing,Remote Work,SaaS|$2.5MM MRR
SavvyCal|Scheduling tool that alleviates the friction of booking meetings|Productivity,SaaS,Web Apps|$458K MRR
Referral Factory|No-code referral marketing tool that helps businesses gain leads|Marketing,SaaS,No-Code|$458K MRR
BambooHR|SaaS platform offering HR solutions for small and medium businesses|SaaS,HR,Small Business|$19.8MM MRR
Clockify|Free time-tracking tool for monitoring productivity|Productivity,Project Management,SaaS|$458K MRR
Designjoy|Subscription-based design service for technology companies and agencies|Design,Subscription Box,SaaS|$258K MRR
Ahrefs|Suite of SEO tools for businesses to enhance online visibility|SaaS,Marketing,Business Intelligence|$8.33MM MRR
Canny|Tool for businesses to gather, organize, and act on user feedback|Business Intelligence,SaaS,Market Research|$280K+ MRR
SmallPDF|Suite of tools that allows users to be more productive and work smarter with documents|Productivity,Web Apps,SaaS|$1.46MM MRR
Browserless|Headless browser service that simplifies web automation tasks for developers|Automation,Developer Tools,Software Development|$131K MRR
Lunch Money|Simple budgeting tool with multi-currency support|Financial Services,Mobile Apps,Travel & Tourism|$34K MRR
Bubble|No-code platform enabling users to build products without technical skills|Software Development,No-Code,Entrepreneurship|$5MM MRR
Plausible Analytics|Privacy-friendly web analytics tool for businesses seeking ethical data insights|Analytics,Data,Open Source|$175K+ MRR
Streak|CRM integrated with Gmail|CRM,Email,Productivity|$833K+ MRR
Balsamiq|Easy-to-use tool to create wireframes|Design,SaaS,Web Apps|$580K MRR
ChartMogul|Subscription analytics tool that helps SaaS companies understand their metrics|Analytics,SaaS,Business Intelligence|$834K MRR
Ghost|Easy-to-use, open-source blogging platform|Blogging Platforms,Web Hosting,SaaS|$580K+ MRR
Trends.vc|Newsletter offering reports on the latest trends in tech and startups|News & Journalism,Market Research,Info Products|$40K+ MRR
Clearbit|API tool that makes it easy to get demographic data about individuals|APIs,Data|$3MM+ MRR
BuySellAds|Marketplace that connects small publishers with advertisers to automate ad sales|Marketplace,Advertising|$1MM+ MRR
MeetEdgar|Social media tool that automates sharing evergreen content for small businesses|Social Media,Automation,Marketing|$300K+ MRR
Zapier|Tool that lets users automate workflows between different apps|Automation,SaaS,Productivity|$20MM+ MRR
Going|Newsletter that shares flight deals to help travelers save money|Travel & Tourism,Email,Shopping|$300K MRR
Baremetrics|Financial metrics tool for SaaS companies|SaaS,Business Intelligence,Automation|$300K+ MRR
Sifter|Simple bug tracking tool designed for non-technical users|Software Development,SaaS,Web Apps|$20K MRR
Sidekiq|Background job processor for Ruby applications|Software Development,Developer Tools,SaaS|$500K MRR
Growth Machine|Agency that creates and manages content marketing for businesses|Content Marketing,Consulting Services,Marketing|$160K+ MRR
Beardbrand|E-commerce brand offering beard care products|E-commerce,Personal Care & Hygiene|$2MM MRR
Postpone|Tool for scheduling social media posts|Social Media,Automation,Content Creation|$30K MRR
Fearless Business|Business coaching service that helps entrepreneurs grow their businesses|Consulting Services,Personal Development|$25K MRR
Threadless|Community-driven marketplace for custom-designed t-shirts|Marketplace,Community,Fashion & Apparel|$5MM MRR
Canva|Online design tool that makes it easy to create and collaborate|Design,Graphic Design,Web Apps|$170MM+ MRR
PyImageSearch|Blog and resource site for learning computer vision with Python|Info Products,Content Creation,Education|$80K+ MRR
Rails Autoscale|Autoscaling tool for Heroku apps|Software Development,Developer Tools,Cloud Computing|$26K MRR
Alitu|Tool that simplifies podcast editing for content creators|Podcasting,Audio,Content Creation|$45K+ MRR
Userlist|SaaS tool that helps companies personalize customer messaging|SaaS,Email|$30K+ MRR
FeedbackPanda|SaaS tool that automates feedback writing for online English teachers|SaaS,Automation,Education|$55K MRR
Place Card Me|Tool that helps people create and print custom wedding place cards|Weddings,Design,Events & Conferencing|$1.6K+ MRR
SegMetrics|SaaS tool that automates marketing data analysis|SaaS,Automation,Data|$100K+ MRR
Webflow|Visual software development platform for building websites|Software Development,Web Apps,Design|$8MM+ MRR
1 Second Everyday|Mobile app that lets users capture one-second daily videos|Mobile Apps,Video,Social Media|$200K+ MRR
Laravel|Framework that simplifies working with PHP|Software Development,Developer Tools|$250K+ MRR
DigsConnect|Marketplace that connects students with landlords for housing|Marketplace,Real Estate,Education|$100K MRR
Cloud Campaign|Social media management tool for marketing agencies to manage campaigns|Social Media,Marketing,SaaS|$25K+ MRR
Contentyze|SaaS tool that uses AI to generate marketing content|Artificial Intelligence (AI),Content Creation,SaaS|$5K+ MRR
Geocodio|Geocoding tool that converts addresses to coordinates and vice versa|APIs,Developer Tools,SaaS|$80K+ MRR
SquadCast|Remote recording tool that provides high-quality audio and video for podcasters|Podcasting,Audio,Video|$250K+ MRR
Acquire|Marketplace that connects buyers and sellers of small startups|Marketplace,Entrepreneurship,Investing|$53K+ MRR
Examine.com|Subscription-based website providing scientifically researched nutrition information|Nutrition,Health & Wellness,Info Products|$83K+ MRR
Hotjar|Analytics tool that helps businesses understand user behavior|Analytics,Business Intelligence,SaaS|$3.33MM+ MRR
Bluetick|Email tool that automates the process of following up with recipients|Email,Productivity,CRM|$100K+ MRR
SEOTesting.com|Tool that helps businesses run and analyze SEO tests for better results|SEO,Automation,Analytics|$18K+ MRR
Astalty|Platform that helps NDIS providers manage their business operations|SaaS,CRM,Health & Wellness|$60K+ MRR
Builder Prime|CRM designed for home improvement contractors|CRM,Home & Garden,SaaS|$80K+ MRR
Tally|Free, intuitive form builder for startups, creators, and small businesses|SaaS,Web Apps,Freemium|$100K MRR
muk mat|High-quality door mat for people who love the outdoors|E-commerce,Home & Garden,Retail|$250K+ MRR
Boot.dev|Gamified platform for learning backend development languages|Education,Software Development,Online Courses|$110K MRR
AI Scout|Directory of AI tools tailored to business needs|Artificial Intelligence (AI),Business Intelligence,Info Products|$8K+ MRR
A Self Guru|Legal template and service provider for entrepreneurs to protect their businesses|Legal,Entrepreneurship,Consulting Services|$83K+ MRR
Stripo|Email design tool that simplifies creating professional emails|Email,Design,Marketing|$300K+ MRR
Zum Rails|SaaS tool that combines open banking with instant payment processing for businesses|SaaS,FinTech|$833K+ MRR
Unicorn Digital Marketing Assistant School|E-learning business teaching women to become marketing assistants|Online Courses,Education,Marketing|$23K MRR
VizyPay|Payment processing service tailored for small businesses in rural areas|FinTech,Financial Services,Local|$1.8MM+ MRR
Grasshopper Labs|AI-driven platform that optimizes transportation and warehousing for businesses|Artificial Intelligence (AI),Transportation,SaaS|$400K MRR
Xtreme Xperience|Service offering supercar driving experiences on racetracks|Entertainment,Sports & Competition|$2.5MM MRR
Baresquare|AI-powered analytics tool that provides actionable insights for e-commerce businesses|Analytics,Artificial Intelligence (AI),E-commerce|$100K MRR
Starter Story|Website showcasing interviews with entrepreneurs about their startup journeys|Entrepreneurship,Content Creation,News & Journalism|$41K+ MRR
Grasshopper|Virtual phone system designed for entrepreneurs to handle calls professionally|SaaS,Telecommunications,Customer Service|$2.5MM+ MRR
Chekkit|Comprehensive customer communication tool for local businesses|Local,SaaS,Customer Relationship Management (CRM)|$166K+ MRR
EaseMyTrip|Online travel platform that doesn't charge convenience fees|Travel & Tourism|$6.4MM MRR
Monite|B2B API fintech platform for finance software services|APIs,FinTech,B2B|$1MM MRR
Upvoty|SaaS tool that streamlines gathering and managing user feedback|SaaS,Consumer Research,Community|$20K+ MRR
ProfitWell|Platform optimizing pricing and payment strategies for businesses|Data,Analytics,Financial Services|$1.8MM+ MRR
Tweet Hunter|SaaS tool that helps users grow and monetize their Twitter audience|SaaS,Social Media,Marketing|$41K+ MRR
ZenMaid|SaaS for maid services to manage schedules, handle communication, and automate tasks|SaaS,Home & Garden,Automation|$130K+ MRR
Headlime|AI-powered tool that generates marketing content|Artificial Intelligence (AI),Copywriting,SaaS|$20K MRR
Substack|Platform for independent writers to publish newsletters and charge for subscriptions|Blogging Platforms,News & Journalism,Content Creation|$1MM MRR
Teachable|SaaS marketplace that allows experts to create and sell online courses|SaaS,Marketplace,Online Courses|$2.1MM+ MRR
CoderPad|Web-based platform for conducting technical interviews with programmers|Hiring,Software Development,Web Apps|$170K+ MRR
Edusign|Platform for managing educational attendance sheets and document signatures|Education,SaaS|$300K MRR
IBISWorld|Global provider of detailed and analytical industry research reports|Market Research,Data,Consulting Services|$1.45MM+ MRR
AppSumo|Daily deals website that curates discounts on software and digital goods|Entrepreneurship,Marketplace,Web Apps|$6.67MM+ MRR
Tettra|Slack-integrated wiki tool for teams to document their knowledge|Knowledge Management,Productivity,SaaS|$25K MRR
Fomo|Tool that boosts website conversions by showing visitors what other customers are buying|Marketing,SaaS,Web Apps|$88K MRR
EditionGuard|Digital rights management service for protecting eBooks from piracy|SaaS,Books & Reading,E-commerce|$25K MRR
Nomad List|Community that helps digital nomads find the best places to live and work remotely|Community,Remote Work,Travel & Tourism|$60K MRR
Gumroad|Streamlined e-commerce platform for digital content creators|E-commerce,Content Creation,Creator Economy|$1.8MM+ MRR
Moonlight|Community for software engineers that connects them with employers|Community,Hiring,Subscription Box|$55K MRR
egghead.io|Site that teaches coding skills through bite-sized screencasts|Education,Online Courses,Web Apps|$308K MRR
HeyGen|Video-creation tool that lets users create talking avatars using AI|Artificial Intelligence (AI),Video,Content Creation|$1.67MM MRR
Stratechery|Newsletter for tech investors that analyzes tech and media news|Newsletter,News & Journalism,Investing|$267K MRR
BuiltWith|Tool that can reveal the tech stack behind any website|Analytics,Market Research,Web Apps|$1.17MM MRR
Devv|AI-powered search engine that helps developers find reliable answers to coding questions|Artificial Intelligence (AI),Search,Developer Tools|$30K MRR
Site Builder Report|Site that reviews website builders and publishes reports on which ones are the best|Reviews,Affiliate & Referral,SEO|$40K MRR
ConvertKit|Email marketing tool tailored to help creators to grow their audiences|Email Marketing,Creator Economy,Blogging Platforms|$3.42MM MRR"""


def parse_mrr(mrr_str):
    """Parse MRR string like '$208K+ MRR' into integer cents."""
    s = mrr_str.replace("MRR","").replace("+","").replace("<","").replace("$","").strip()
    if "MM" in s:
        return int(float(s.replace("MM","")) * 1_000_000)
    elif "K" in s:
        return int(float(s.replace("K","")) * 1_000)
    else:
        return int(float(s))

def parse_ideas():
    ideas = []
    for line in RAW.strip().split("\n"):
        parts = line.split("|")
        if len(parts) != 4:
            continue
        name, desc, cats, mrr = parts
        ideas.append({
            "name": name.strip(),
            "description": desc.strip(),
            "categories": [c.strip() for c in cats.split(",")],
            "mrr_usd": parse_mrr(mrr),
            "mrr_raw": mrr.strip(),
        })
    return ideas

def upload_to_postgres(ideas):
    conn = psycopg2.connect(host="localhost", port=5432, user="admin", password="admin", dbname="feast_offline")
    cur = conn.cursor()
    now = datetime.now(timezone.utc)

    # Create table
    cur.execute("""
    CREATE TABLE IF NOT EXISTS indiehackers_ideas (
        id              SERIAL PRIMARY KEY,
        name            VARCHAR(128) NOT NULL,
        description     TEXT NOT NULL,
        categories      TEXT[] NOT NULL,
        category_primary VARCHAR(64),
        mrr_usd         INTEGER,
        mrr_raw         VARCHAR(32),
        source          VARCHAR(32) DEFAULT 'indiehackers',
        scraped_at      TIMESTAMPTZ NOT NULL,
        event_timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        created_timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        UNIQUE(name, description)
    );
    """)

    # Feature table for feast
    cur.execute("""
    CREATE TABLE IF NOT EXISTS indiehackers_idea_features (
        idea_id             VARCHAR(128) PRIMARY KEY,
        name                VARCHAR(128) NOT NULL,
        description         TEXT NOT NULL,
        category_primary    VARCHAR(64),
        category_count      INTEGER,
        mrr_usd             INTEGER,
        description_length  INTEGER,
        has_ai              BOOLEAN,
        is_saas             BOOLEAN,
        is_marketplace      BOOLEAN,
        event_timestamp     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        created_timestamp   TIMESTAMPTZ NOT NULL DEFAULT NOW()
    );
    """)

    # Insert ideas
    idea_rows = []
    feature_rows = []
    for i, idea in enumerate(ideas):
        cats = idea["categories"]
        primary = cats[0] if cats else None
        idea_rows.append((
            idea["name"], idea["description"], cats, primary,
            idea["mrr_usd"], idea["mrr_raw"], "indiehackers", now, now, now,
        ))
        idea_id = f"ih_{i+1:03d}"
        cat_str = " ".join(cats).lower()
        feature_rows.append((
            idea_id, idea["name"], idea["description"], primary, len(cats),
            idea["mrr_usd"], len(idea["description"]),
            "artificial intelligence" in cat_str or "ai" in cat_str,
            "saas" in cat_str,
            "marketplace" in cat_str,
            now, now,
        ))

    execute_values(cur, """
        INSERT INTO indiehackers_ideas (name, description, categories, category_primary,
            mrr_usd, mrr_raw, source, scraped_at, event_timestamp, created_timestamp)
        VALUES %s ON CONFLICT (name, description) DO UPDATE SET mrr_usd = EXCLUDED.mrr_usd
    """, idea_rows)

    execute_values(cur, """
        INSERT INTO indiehackers_idea_features (idea_id, name, description, category_primary,
            category_count, mrr_usd, description_length, has_ai, is_saas, is_marketplace,
            event_timestamp, created_timestamp)
        VALUES %s ON CONFLICT (idea_id) DO UPDATE SET mrr_usd = EXCLUDED.mrr_usd
    """, feature_rows)

    conn.commit()

    # Verify
    cur.execute("SELECT COUNT(*) FROM indiehackers_ideas")
    print(f"indiehackers_ideas: {cur.fetchone()[0]} rows")
    cur.execute("SELECT COUNT(*) FROM indiehackers_idea_features")
    print(f"indiehackers_idea_features: {cur.fetchone()[0]} rows")
    cur.execute("SELECT AVG(mrr_usd), MIN(mrr_usd), MAX(mrr_usd) FROM indiehackers_ideas")
    avg, mn, mx = cur.fetchone()
    print(f"MRR range: ${mn:,} - ${mx:,} (avg ${avg:,.0f})")
    cur.execute("SELECT COUNT(*) FILTER (WHERE has_ai), COUNT(*) FILTER (WHERE is_saas), COUNT(*) FILTER (WHERE is_marketplace) FROM indiehackers_idea_features")
    ai, saas, mp = cur.fetchone()
    print(f"AI: {ai}, SaaS: {saas}, Marketplace: {mp}")

    cur.close()
    conn.close()

def save_json(ideas):
    out = Path("data/indiehackers_ideas.json")
    with open(out, "w") as f:
        json.dump(ideas, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(ideas)} ideas to {out}")

def main():
    ideas = parse_ideas()
    print(f"Parsed {len(ideas)} ideas from Indie Hackers")
    save_json(ideas)
    print("\nUploading to Postgres...")
    upload_to_postgres(ideas)
    print("\nDone.")

if __name__ == "__main__":
    main()
