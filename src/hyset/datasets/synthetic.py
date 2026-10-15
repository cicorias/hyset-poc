"""Synthetic tool library + labeled queries with planted set-level structure.

The generator is built to exhibit the failure modes that motivate HySet:

- **Lexical decoys.** Popular capabilities get look-alike tools (e.g. a fare-deals API
  next to the real flight search) that are never in a ground-truth set. Independent
  scorers rank them highly; a set scorer learns that a second flight API adds nothing.
- **Implicit sub-tasks.** Some sub-tasks are phrased with little lexical overlap
  ("tell me what to pack" -> weather). They are recoverable from co-invocation
  patterns rather than wording.
- **Cardinality-dependent co-invocation.** Scenarios are weighted so that, e.g., weather
  and currency are common together in 4-5 tool trip-planning sets but rare as a pair.
- **Novel compositions.** A fraction of queries combine random capabilities, so the
  test split contains tool sets never seen in training (flagged ``meta.unseen_set``).

Every tool carries a ``capability`` label, which the simulated reward uses.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field

from ..data import Dataset, Example, Tool, ToolLibrary


@dataclass
class Capability:
    key: str
    domain: str
    tool: tuple[str, str]  # (name, description) of the one annotated tool
    explicit: list[str]
    implicit: list[str] = field(default_factory=list)
    decoys: list[tuple[str, str]] = field(default_factory=list)  # never ground truth


C = Capability
CAPABILITIES: list[Capability] = [
    # ---------------------------------------------------------------- travel
    C(
        "flight_search",
        "travel",
        (
            "SkyRoute Flight Search",
            "Search one-way and round-trip flights between airports; returns itineraries, airlines, stops and fares.",
        ),
        [
            "find round-trip flights from {city} to {city2}",
            "search flights to {city2} {date}",
            "look up airfares from {city} to {city2}",
        ],
        ["I need to get from {city} to {city2} {date}"],
        [
            (
                "CheapFlights Deals",
                "Browse cheap flights deals and discounted round-trip airfare promotions from popular cities.",
            ),
            (
                "FlightTracker Live",
                "Track live flights: status, delays, gates and arrival times for flights by flight number.",
            ),
            ("Airline Seat Maps", "Show seat maps and seat availability for airline flights."),
        ],
    ),
    C(
        "hotel_booking",
        "travel",
        (
            "StayFinder Hotels",
            "Search and book hotel rooms by city or neighborhood with prices, ratings and availability.",
        ),
        [
            "book a hotel near {landmark}",
            "find a hotel in {city2} for {nights} nights",
            "reserve a room close to {landmark}",
        ],
        ["I need somewhere to sleep near {landmark}"],
        [("Hotel Reviews Digest", "Read guest reviews and photos of hotels in a city.")],
    ),
    C(
        "car_rental",
        "travel",
        ("DriveNow Car Rental", "Compare and reserve rental cars at airports and city locations."),
        ["rent a car at the {city2} airport", "reserve a rental car for {nights} days"],
        ["I want to drive myself around once I land"],
    ),
    C(
        "flight_status",
        "travel",
        (
            "AirStatus Flight Status",
            "Get real-time status of a specific flight by airline and flight number: delays, gate and terminal.",
        ),
        ["check whether flight {flight} is delayed", "get the gate and status of flight {flight}"],
        ["is my {flight} plane on time"],
    ),
    C(
        "visa_requirements",
        "travel",
        ("VisaCheck", "Look up passport and visa entry requirements between countries."),
        ["check if I need a visa for {country}", "look up entry requirements for {country}"],
        ["will they let me in to {country} with my passport"],
    ),
    # ---------------------------------------------------------------- weather / env
    C(
        "weather_forecast",
        "weather",
        (
            "WeatherNow Forecast",
            "Daily and hourly weather forecasts: temperature, precipitation and wind for any city and date range.",
        ),
        [
            "check the weather in {city2} for those dates",
            "get the forecast for {city2} {date}",
            "tell me if it will rain in {city2}",
        ],
        ["tell me what clothes to pack", "let me know if I should bring an umbrella"],
        [
            (
                "Climate History Archive",
                "Historical climate records and average weather statistics by month.",
            )
        ],
    ),
    C(
        "air_quality",
        "weather",
        (
            "AirSense Air Quality",
            "Current air quality index, pollen and pollution levels for a location.",
        ),
        ["check the air quality in {city2}", "get the pollen count in {city}"],
        ["is it safe for my asthma to jog outside in {city}"],
    ),
    # ---------------------------------------------------------------- finance
    C(
        "currency_convert",
        "finance",
        (
            "FXRates Currency Converter",
            "Convert amounts between currencies using live exchange rates.",
        ),
        [
            "convert my {amount} {cur} budget into {cur2}",
            "convert {amount} {cur} to {cur2}",
            "get the {cur}/{cur2} exchange rate",
        ],
        [
            "my budget is {amount} {cur}, how far does that go locally",
            "figure out how much spending money that is over there",
        ],
        [
            (
                "Forex Trading Signals",
                "Trading signals and charts for foreign exchange currency pairs.",
            )
        ],
    ),
    C(
        "stock_quote",
        "finance",
        (
            "MarketQuote Stocks",
            "Real-time and historical stock quotes, market cap and price charts by ticker.",
        ),
        ["get the stock price of {ticker}", "show me {ticker}'s share price this week"],
        ["how is {ticker} doing on the market today"],
        [("Stock Screener Pro", "Screen stocks by sector, dividend yield and valuation ratios.")],
    ),
    C(
        "crypto_price",
        "finance",
        (
            "CoinPulse Crypto Prices",
            "Live cryptocurrency prices, volume and market data for coins and tokens.",
        ),
        ["get the current price of {coin}", "check {coin}'s 24h price change"],
        ["is {coin} up or down today"],
    ),
    C(
        "tax_calculator",
        "finance",
        (
            "TaxEstimate Calculator",
            "Estimate income tax and sales tax for a given amount and region.",
        ),
        ["estimate the tax on {amount} {cur}", "calculate sales tax for a {amount} {cur} purchase"],
    ),
    # ---------------------------------------------------------------- food
    C(
        "restaurant_search",
        "food",
        (
            "TableFinder Restaurants",
            "Find restaurants by cuisine, location and price, and check table availability.",
        ),
        [
            "find a {cuisine} restaurant in {city}",
            "recommend a {cuisine} place for dinner near {landmark}",
        ],
        ["where should we eat tonight near {landmark}"],
        [
            (
                "Food Blog Highlights",
                "Curated food blog articles about dining trends and chef interviews.",
            )
        ],
    ),
    C(
        "recipe_search",
        "food",
        ("RecipeBox", "Search recipes by dish, ingredients or diet with steps and cooking times."),
        ["find a recipe for {dish}", "give me a {diet} recipe for {dish}"],
        ["what can I cook with what's in my fridge"],
    ),
    C(
        "nutrition_facts",
        "food",
        ("NutriFacts", "Calories, macronutrients and allergens for foods and recipes."),
        ["get the calories and macros for {dish}", "check the nutrition facts of {dish}"],
        ["is {dish} healthy enough for my diet"],
    ),
    C(
        "grocery_delivery",
        "food",
        ("FreshCart Grocery Delivery", "Order groceries online for same-day home delivery."),
        ["order the ingredients for delivery", "get groceries delivered to {address}"],
        ["have the stuff brought to my door"],
    ),
    # ---------------------------------------------------------------- maps / transit
    C(
        "geocode",
        "maps",
        ("GeoLocate", "Convert addresses to coordinates and coordinates to addresses."),
        ["get the coordinates of {address}", "geocode {address}"],
    ),
    C(
        "directions",
        "maps",
        (
            "RoutePlanner Directions",
            "Turn-by-turn driving, walking and cycling directions between places.",
        ),
        ["get directions from {address} to {landmark}", "how do I walk to {landmark}"],
        ["show me the way there"],
        [("Map Tiles Renderer", "Render map tiles and static map images for websites.")],
    ),
    C(
        "traffic_info",
        "maps",
        ("TrafficWatch", "Live traffic conditions, incidents and travel-time estimates on roads."),
        ["check traffic on the way to {landmark}", "is there traffic near {city} right now"],
        ["will I be stuck in a jam getting there"],
    ),
    C(
        "transit_schedule",
        "maps",
        ("TransitTimes", "Public transport schedules for buses, trains and subways."),
        ["get train times from {city} to {city2}", "find the next subway to {landmark}"],
    ),
    # ---------------------------------------------------------------- events / sports
    C(
        "concert_finder",
        "events",
        ("GigGuide Concerts", "Find upcoming concerts and live music events by artist or city."),
        ["find {artist} concerts in {city}", "see when {artist} is touring near {city}"],
        ["I want to see {artist} live"],
    ),
    C(
        "ticket_booking",
        "events",
        ("TicketHub", "Buy tickets for concerts, sports games and shows."),
        ["buy two tickets", "book tickets for the show", "get seats for the game"],
        ["grab us a pair of seats"],
        [("Ticket Resale Prices", "Track ticket resale price trends on secondary markets.")],
    ),
    C(
        "sports_scores",
        "events",
        ("ScoreBoard Live", "Live and final scores, box scores and standings for major sports."),
        ["get the latest {team} score", "show last night's {team} result"],
        ["did {team} win"],
    ),
    C(
        "sports_schedule",
        "events",
        ("FixtureList Schedules", "Upcoming fixtures and game schedules for sports teams."),
        ["find {team}'s next home game", "get {team}'s schedule for {date}"],
        ["when do {team} play next"],
    ),
    # ---------------------------------------------------------------- media
    C(
        "song_search",
        "media",
        ("TuneFinder Music Search", "Search songs, albums and artists with streaming links."),
        ["find {artist}'s most popular songs", "search for the song {song}"],
    ),
    C(
        "lyrics_lookup",
        "media",
        ("LyricVault", "Retrieve full song lyrics by title and artist."),
        ["get the lyrics of {song}", "show me the words to {song}"],
        ["I want to sing along"],
    ),
    C(
        "movie_info",
        "media",
        ("CineData Movies", "Movie details: cast, ratings, plot and showtimes."),
        ["get the cast and rating of {movie}", "find showtimes for {movie} in {city}"],
    ),
    C(
        "podcast_search",
        "media",
        ("PodSearch", "Search podcasts and episodes by topic or host."),
        ["find podcasts about {topic}", "search for an episode on {topic}"],
    ),
    # ---------------------------------------------------------------- comms / productivity
    C(
        "send_email",
        "comms",
        ("MailSender", "Compose and send emails to one or more recipients."),
        ["email the details to {person}", "send {person} an email with the plan"],
        ["let {person} know by mail"],
    ),
    C(
        "send_sms",
        "comms",
        ("TextBlast SMS", "Send SMS text messages to phone numbers."),
        ["text {person} the details", "send an SMS to {person}"],
    ),
    C(
        "calendar_event",
        "comms",
        ("CalendarSync", "Create, update and list calendar events and reminders."),
        [
            "add it to my calendar",
            "create a calendar event for {date}",
            "set a reminder for {date}",
        ],
        ["make sure I don't forget"],
    ),
    C(
        "translate_text",
        "comms",
        ("LinguaTranslate", "Translate text between more than 100 languages."),
        ["translate a few phrases into {lang}", "translate my message to {lang}"],
        ["help me talk to the locals in {lang}"],
    ),
    # ---------------------------------------------------------------- commerce
    C(
        "product_search",
        "commerce",
        (
            "ShopSearch Products",
            "Search products across online stores with specs and availability.",
        ),
        ["find a {product}", "search for a good {product}"],
        ["I'm in the market for a {product}"],
        [("Product Unboxing Videos", "Watch unboxing and review videos for consumer products.")],
    ),
    C(
        "price_compare",
        "commerce",
        ("PriceWise Compare", "Compare prices for the same product across retailers."),
        ["compare prices for the {product}", "find the cheapest store for a {product}"],
        ["make sure I'm not overpaying"],
    ),
    C(
        "order_tracking",
        "commerce",
        ("ParcelTrack Orders", "Track package and order shipping status by tracking number."),
        ["track my order {order}", "where is package {order}"],
    ),
    C(
        "coupon_finder",
        "commerce",
        ("DealCoupons", "Find coupon codes and discounts for online stores."),
        ["find a coupon code", "look for a discount code for the purchase"],
    ),
    # ---------------------------------------------------------------- information
    C(
        "news_headlines",
        "info",
        ("NewsWire Headlines", "Top news headlines and articles by topic, source or country."),
        ["get the latest news about {topic}", "show today's headlines on {topic}"],
        ["what's going on with {topic}"],
    ),
    C(
        "wikipedia_summary",
        "info",
        ("WikiBrief", "Short encyclopedia summaries of people, places and concepts."),
        ["give me a short summary of {topic}", "look up background on {topic}"],
    ),
    C(
        "sentiment_analysis",
        "info",
        ("MoodMeter Sentiment", "Analyze the sentiment and emotion of text."),
        [
            "analyze the sentiment of those articles",
            "tell me if the coverage is positive or negative",
        ],
    ),
    C(
        "text_summarize",
        "info",
        ("TLDR Summarizer", "Summarize long documents and articles into key points."),
        ["summarize the articles into bullet points", "condense it into a short summary"],
    ),
    # ---------------------------------------------------------------- health
    C(
        "symptom_checker",
        "health",
        ("SymptomScout", "Check symptoms and get possible conditions and triage advice."),
        ["check what could cause {symptom}", "assess my {symptom}"],
        ["I've been feeling off with {symptom}"],
    ),
    C(
        "doctor_finder",
        "health",
        (
            "MedFinder Doctors",
            "Find doctors and clinics by specialty and location with appointment slots.",
        ),
        ["find a doctor in {city} who can see me", "book a clinic appointment in {city}"],
    ),
    C(
        "pharmacy_finder",
        "health",
        ("PharmaNear", "Locate nearby pharmacies with opening hours."),
        ["find an open pharmacy near {address}", "locate a pharmacy in {city}"],
    ),
]
del C

# Workflows over capabilities with sampling weights. Weather+currency is common inside
# large trip sets but rare as a pair: cardinality-dependent co-invocation.
SCENARIOS: list[tuple[tuple[str, ...], float]] = [
    (("flight_search", "hotel_booking", "weather_forecast", "currency_convert"), 6.0),
    (
        (
            "flight_search",
            "hotel_booking",
            "weather_forecast",
            "currency_convert",
            "translate_text",
        ),
        2.5,
    ),
    (("flight_search", "hotel_booking", "car_rental", "weather_forecast", "currency_convert"), 2.0),
    (("flight_search", "hotel_booking"), 3.0),
    (("flight_search", "hotel_booking", "car_rental"), 2.0),
    (("flight_search", "visa_requirements", "currency_convert"), 1.0),
    (("flight_status", "car_rental"), 1.0),
    (("flight_status", "send_sms"), 1.0),
    (("weather_forecast", "currency_convert"), 0.15),
    (("weather_forecast",), 2.0),
    (("weather_forecast", "air_quality"), 1.0),
    (("currency_convert",), 1.5),
    (("stock_quote", "news_headlines", "sentiment_analysis"), 2.0),
    (("stock_quote", "news_headlines", "sentiment_analysis", "text_summarize"), 1.0),
    (("stock_quote", "currency_convert"), 1.0),
    (("crypto_price", "currency_convert"), 1.5),
    (("crypto_price", "news_headlines"), 1.0),
    (("tax_calculator", "currency_convert"), 0.8),
    (("restaurant_search", "directions"), 2.0),
    (("restaurant_search", "directions", "traffic_info"), 1.0),
    (("restaurant_search", "calendar_event", "send_sms"), 1.0),
    (("recipe_search", "nutrition_facts"), 2.0),
    (("recipe_search", "grocery_delivery", "nutrition_facts", "calendar_event"), 1.5),
    (("recipe_search", "grocery_delivery"), 1.0),
    (("geocode", "directions", "traffic_info"), 1.2),
    (("directions", "transit_schedule"), 1.0),
    (("concert_finder", "ticket_booking", "calendar_event"), 2.0),
    (("concert_finder", "ticket_booking", "calendar_event", "directions", "weather_forecast"), 1.0),
    (("sports_schedule", "ticket_booking", "directions", "weather_forecast"), 1.5),
    (("sports_scores",), 1.0),
    (("sports_scores", "sports_schedule"), 1.0),
    (("song_search", "lyrics_lookup"), 1.5),
    (("movie_info", "calendar_event", "send_email"), 1.0),
    (("podcast_search",), 0.8),
    (("send_email", "calendar_event"), 1.5),
    (("translate_text", "send_email"), 1.0),
    (("product_search", "price_compare", "coupon_finder"), 1.5),
    (("product_search", "price_compare", "currency_convert"), 1.0),
    (("order_tracking", "send_sms"), 0.8),
    (("order_tracking",), 0.8),
    (("news_headlines", "text_summarize", "send_email"), 1.2),
    (("wikipedia_summary", "translate_text"), 0.8),
    (("symptom_checker", "doctor_finder", "directions"), 1.2),
    (("symptom_checker", "pharmacy_finder"), 1.0),
    (("doctor_finder", "calendar_event", "traffic_info"), 0.8),
]

SLOTS: dict[str, list[str]] = {
    "city": ["Chicago", "Boston", "Seattle", "Denver", "Austin", "Toronto", "London", "Berlin"],
    "city2": ["Tokyo", "Paris", "Rome", "Lisbon", "Seoul", "Mexico City", "Bangkok", "Sydney"],
    "landmark": [
        "Shinjuku",
        "the Eiffel Tower",
        "the Colosseum",
        "the old town",
        "the convention center",
        "the harbor",
    ],
    "date": [
        "next month",
        "next weekend",
        "from June 3 to June 8",
        "over Thanksgiving",
        "this Friday",
        "in early May",
    ],
    "nights": ["three", "four", "five", "seven"],
    "amount": ["2,000", "500", "1,200", "3,500", "750"],
    "cur": ["USD", "CAD", "GBP", "EUR"],
    "cur2": ["yen", "euros", "won", "baht", "pesos", "Australian dollars"],
    "country": ["Japan", "Brazil", "India", "Vietnam", "Kenya"],
    "flight": ["UA 881", "DL 15", "AA 100", "NH 9"],
    "ticker": ["AAPL", "NVDA", "TSLA", "MSFT", "AMZN"],
    "coin": ["Bitcoin", "Ethereum", "Solana"],
    "cuisine": ["Thai", "Italian", "sushi", "vegan", "Mexican"],
    "dish": ["lasagna", "pad thai", "banana bread", "chili", "ramen"],
    "diet": ["vegetarian", "gluten-free", "keto", "low-sodium"],
    "address": ["221 Main Street", "my office", "the train station", "my apartment"],
    "artist": ["Taylor Swift", "Coldplay", "Bad Bunny", "Billie Eilish"],
    "team": ["the Cubs", "the Lakers", "Arsenal", "the Packers"],
    "song": ["Yellow", "Bohemian Rhapsody", "Shake It Off"],
    "movie": ["Dune", "Inception", "Oppenheimer"],
    "topic": [
        "electric vehicles",
        "the election",
        "AI regulation",
        "climate policy",
        "the World Cup",
    ],
    "person": ["my wife", "my boss", "Alex", "the team"],
    "lang": ["Japanese", "Spanish", "French", "Korean"],
    "product": ["noise-cancelling headphones", "standing desk", "espresso machine", "4K monitor"],
    "order": ["#A1932", "#77310", "1Z999AA10"],
    "symptom": ["a persistent cough", "migraines", "a sore throat"],
}

OPENERS = ["", "", "Hi! ", "Can you help me? ", "Quick request: ", "Please help. "]
FILLER_ADJ = [
    "Quantum",
    "Open",
    "Smart",
    "Rapid",
    "Global",
    "Bright",
    "Nimble",
    "Secure",
    "Micro",
    "Hyper",
]
FILLER_NOUN = [
    ("Invoice", "invoices"), ("Barcode", "barcodes"), ("Domain WHOIS", "domain registrations"),
    ("QR Code", "QR codes"), ("Email Validator", "email address validity"), ("Joke", "random jokes"),
    ("Horoscope", "daily horoscopes"), ("Pet Adoption", "adoptable pets"), ("License Plate", "vehicle plates"),
    ("Real Estate", "property listings"), ("Job Board", "job postings"), ("Trivia", "trivia questions"),
    ("Password", "password strength"), ("IP Lookup", "IP geolocation"), ("Image Resize", "image resizing"),
    ("Font", "web fonts"), ("Color Palette", "color palettes"), ("Unit Converter", "unit conversions"),
    ("Dictionary", "word definitions"), ("Quote", "famous quotes"),
]  # fmt: skip
FILLER_VERB = ["lookup", "generation", "validation", "search", "analytics", "monitoring"]


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def build_library(n_filler: int = 150, seed: int = 0) -> ToolLibrary:
    rng = random.Random(seed)
    tools: list[Tool] = []
    for cap in CAPABILITIES:
        name, desc = cap.tool
        tools.append(Tool(_slug(name), name, desc, cap.domain, cap.key))
        for dname, ddesc in cap.decoys:
            tools.append(Tool(_slug(dname), dname, ddesc, cap.domain, f"decoy:{_slug(dname)}"))
    seen = {t.id for t in tools}
    max_filler = len(FILLER_ADJ) * len(FILLER_NOUN) * len(FILLER_VERB)
    added = 0
    while added < min(n_filler, max_filler):
        adj = rng.choice(FILLER_ADJ)
        noun, what = rng.choice(FILLER_NOUN)
        verb = rng.choice(FILLER_VERB)
        name = f"{adj} {noun} {verb.title()} API"
        tid = _slug(name)
        if tid in seen:
            continue
        seen.add(tid)
        desc = f"{verb.capitalize()} service for {what}, with a simple REST interface."
        tools.append(Tool(tid, name, desc, "misc", f"filler:{tid}"))
        added += 1
    return ToolLibrary(tools)


def _fill(template: str, slots: dict[str, str]) -> str:
    return re.sub(r"\{(\w+)\}", lambda m: slots[m.group(1)], template)


def _join(phrases: list[str], rng: random.Random) -> str:
    if len(phrases) == 1:
        body = phrases[0]
    elif rng.random() < 0.7:
        body = ", ".join(phrases[:-1]) + (", and " if len(phrases) > 2 else " and ") + phrases[-1]
    else:
        body = ". Also, ".join(phrases)
    return body[0].upper() + body[1:] + "."


def make_query(caps: list[Capability], rng: random.Random, p_implicit: float) -> str:
    slots = {k: rng.choice(v) for k, v in SLOTS.items()}
    phrases = []
    for cap in caps:
        pool = cap.implicit if cap.implicit and rng.random() < p_implicit else cap.explicit
        phrases.append(_fill(rng.choice(pool), slots))
    if len(phrases) > 2 and rng.random() < 0.3:  # light reordering
        head, tail = phrases[:1], phrases[1:]
        rng.shuffle(tail)
        phrases = head + tail
    context = ""
    if caps[0].key == "flight_search" and rng.random() < 0.5:
        context = _fill("I'm flying from {city} to {city2} {date}. ", slots)
    return rng.choice(OPENERS) + context + _join(phrases, rng)


def generate(
    n_train: int = 4000,
    n_val: int = 400,
    n_test: int = 600,
    n_filler: int = 150,
    p_random: float = 0.12,
    p_implicit: float = 0.35,
    seed: int = 0,
) -> Dataset:
    rng = random.Random(seed)
    library = build_library(n_filler, seed)
    by_key = {c.key: c for c in CAPABILITIES}
    primary = {c.key: _slug(c.tool[0]) for c in CAPABILITIES}
    scen, weights = zip(*SCENARIOS, strict=True)

    def sample() -> tuple[str, ...]:
        if rng.random() < p_random:
            m = rng.choice((2, 3, 3, 4))
            return tuple(rng.sample(sorted(by_key), m))
        return rng.choices(scen, weights=weights)[0]

    def make(split: str, n: int) -> list[Example]:
        out = []
        for i in range(n):
            keys = sample()
            caps = [by_key[k] for k in keys]
            query = make_query(caps, rng, p_implicit)
            out.append(
                Example(
                    qid=f"{split}-{i:05d}",
                    query=query,
                    tools=tuple(primary[k] for k in keys),
                    meta={"capabilities": list(keys)},
                )
            )
        return out

    splits = {
        "train": make("train", n_train),
        "val": make("val", n_val),
        "test": make("test", n_test),
    }
    train_sets = {frozenset(e.tools) for e in splits["train"]}
    for name in ("val", "test"):
        for e in splits[name]:
            e.meta["unseen_set"] = frozenset(e.tools) not in train_sets
    meta = {
        "source": "synthetic",
        "seed": seed,
        "n_tools": len(library),
        "n_capabilities": len(CAPABILITIES),
        "params": {
            "n_train": n_train, "n_val": n_val, "n_test": n_test, "n_filler": n_filler,
            "p_random": p_random, "p_implicit": p_implicit,
        },
    }  # fmt: skip
    ds = Dataset(library, splits, meta)
    ds.validate()
    return ds


# The motivating example from the paper's introduction; GT = {flight, hotel, weather, currency}.
PAPER_EXAMPLE = (
    "I am flying from Chicago to Tokyo for five days next month. Find round-trip flights, "
    "book a hotel near Shinjuku, check the weather for those dates, and convert my 2,000 USD "
    "budget into yen"
)
