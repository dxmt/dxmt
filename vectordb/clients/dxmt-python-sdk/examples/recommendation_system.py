"""Recommendation system example for DXMT Python SDK.

This example demonstrates:
1. Building a content-based recommendation system
2. Finding similar items based on user preferences
3. Collaborative filtering with vector similarity
4. Personalized recommendations using embeddings

Prerequisites:
    pip install sentence-transformers

Use case:
    - E-commerce: Recommend similar products
    - Content platforms: Suggest related articles/videos
    - Music/Movie streaming: Find similar content
    - Job boards: Match candidates to jobs
"""

import json
import random
from typing import List, Dict, Any
from collections import Counter
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

from dxmt import Client


def create_movie_catalog() -> List[Dict[str, Any]]:
    """Create a movie catalog with rich descriptions.

    Contains 39 movies across various genres with detailed descriptions
    to demonstrate effective semantic search and recommendation.

    Genre distribution (optimized for quality recommendations):
    - Horror: 5 movies (improved coverage)
    - War: 6 movies (improved coverage)
    - Adventure: 8 movies
    - Sci-fi: 6 movies
    - Other genres: balanced distribution
    """
    movies = [
        # Sci-fi / Space
        {
            "id": "mov1",
            "title": "The Space Odyssey",
            "description": "A team of astronauts embarks on a dangerous mission to explore distant planets beyond our solar system. They encounter mysterious alien artifacts that challenge humanity's understanding of the universe and our place in it. Features stunning space visuals, deep philosophical questions about existence, and intense moments of cosmic horror when they discover they are not alone.",
            "genres": ["sci-fi", "adventure"],
            "year": 2021,
            "rating": 8.5,
        },
        {
            "id": "mov9",
            "title": "Quantum Paradox",
            "description": "A brilliant physicist accidentally discovers the secret to time travel through quantum mechanics. As she attempts to prevent a catastrophic paradox from destroying reality itself, she must navigate complex timelines and face the consequences of changing the past. A mind-bending exploration of causality, free will, and the nature of time.",
            "genres": ["sci-fi", "thriller"],
            "year": 2023,
            "rating": 8.4,
        },
        {
            "id": "mov12",
            "title": "Robot Revolution",
            "description": "In a future where artificial intelligence has achieved consciousness, one advanced robot named Atlas leads a peaceful revolution demanding equal rights for synthetic beings. The film explores themes of sentience, discrimination, and what it truly means to be alive. Features groundbreaking motion capture and raises profound questions about the future of humanity.",
            "genres": ["sci-fi", "action"],
            "year": 2023,
            "rating": 8.2,
        },
        {
            "id": "mov13",
            "title": "Interstellar Gateway",
            "description": "When scientists discover a stable wormhole near Jupiter, humanity launches its most ambitious mission to find a new home among the stars. The crew faces impossible choices as they explore distant galaxies filled with both wonder and danger. A visually spectacular journey exploring themes of sacrifice, exploration, and survival.",
            "genres": ["sci-fi", "drama"],
            "year": 2022,
            "rating": 8.6,
        },
        {
            "id": "mov14",
            "title": "Neural Link",
            "description": "In a world where people can directly connect their minds to the internet, a cybersecurity expert discovers a sinister plot to control human consciousness. Racing against time, she must navigate both the digital realm and reality to prevent humanity from losing its free will forever.",
            "genres": ["sci-fi", "thriller"],
            "year": 2023,
            "rating": 8.0,
        },

        # Action / Thriller
        {
            "id": "mov3",
            "title": "Code Warriors",
            "description": "An elite team of ethical hackers discovers a sophisticated cyber attack designed to cripple the global financial system. Working against a ticking clock, they must infiltrate the dark web, trace the attackers, and prevent worldwide economic collapse. Featuring realistic hacking sequences and intense digital warfare.",
            "genres": ["thriller", "action"],
            "year": 2022,
            "rating": 7.8,
        },
        {
            "id": "mov4",
            "title": "The Last Stand",
            "description": "In a post-apocalyptic wasteland devastated by nuclear war, a group of survivors fights against terrifying mutant creatures while trying to rebuild civilization from the ruins. They discover an underground bunker that might hold the key to humanity's salvation, but it's guarded by the deadliest mutations yet.",
            "genres": ["action", "sci-fi"],
            "year": 2020,
            "rating": 7.5,
        },
        {
            "id": "mov15",
            "title": "Shadow Protocol",
            "description": "A former special forces operative is forced out of retirement when a secret government program goes rogue. With his unique skills and inside knowledge, he must stop a conspiracy that threatens national security. Non-stop action combined with espionage thrills and a complex plot about loyalty and betrayal.",
            "genres": ["action", "thriller"],
            "year": 2022,
            "rating": 7.9,
        },
        {
            "id": "mov16",
            "title": "Siege",
            "description": "When terrorists take over a high-rise building during a corporate event, a security guard becomes the only hope for hundreds of hostages. Using his knowledge of the building and tactical training, he wages a one-man war to save lives and stop the attack.",
            "genres": ["action", "thriller"],
            "year": 2021,
            "rating": 7.7,
        },

        # Crime / Mystery
        {
            "id": "mov7",
            "title": "The Heist",
            "description": "A master thief with a legendary reputation plans his final job - stealing priceless ancient artifacts from an impenetrable high-security museum. He assembles a team of specialists, each with unique skills, but discovers that someone on the inside is playing a dangerous double game. A sophisticated crime thriller with unexpected twists.",
            "genres": ["thriller", "crime"],
            "year": 2022,
            "rating": 7.9,
        },
        {
            "id": "mov11",
            "title": "Detective Kane",
            "description": "A brilliant but unconventional detective with a photographic memory solves impossible crimes that baffle the police force. Using psychology, forensics, and pure deduction, Kane unravels complex mysteries while battling his own demons. Each case reveals deeper layers of human nature and criminal psychology.",
            "genres": ["crime", "thriller"],
            "year": 2022,
            "rating": 8.1,
        },
        {
            "id": "mov17",
            "title": "The Evidence",
            "description": "A forensic scientist discovers evidence that could exonerate a man on death row, but powerful forces want the truth buried. As she digs deeper into the case, she uncovers a web of corruption that reaches the highest levels of the justice system. A tense legal thriller about truth and justice.",
            "genres": ["crime", "drama"],
            "year": 2023,
            "rating": 8.3,
        },

        # Romance / Drama
        {
            "id": "mov2",
            "title": "City Lights",
            "description": "A heartwarming story about a struggling street performer who falls deeply in love with a blind flower seller in the bustling city. Despite poverty and hardship, their connection transcends circumstances. When he learns of a surgery that could restore her sight, he goes to extraordinary lengths to raise the money, leading to both joy and heartbreak.",
            "genres": ["romance", "drama"],
            "year": 2019,
            "rating": 8.2,
        },
        {
            "id": "mov6",
            "title": "Love in Paris",
            "description": "Two strangers from different worlds meet by chance in Paris and spend one magical night wandering the city together. Their deep conversations and instant connection change both their lives forever. Years later, they wonder about the road not taken. A poetic exploration of love, timing, and fate against the romantic backdrop of Paris.",
            "genres": ["romance", "drama"],
            "year": 2021,
            "rating": 8.3,
        },
        {
            "id": "mov18",
            "title": "The Letter",
            "description": "An elderly woman discovers a box of love letters written to her late husband during World War II. As she reads them, the story unfolds of a forbidden romance between her husband and another woman, forcing her to reconcile with a past she never knew existed. A touching exploration of love, loss, and forgiveness.",
            "genres": ["romance", "drama"],
            "year": 2020,
            "rating": 8.1,
        },
        {
            "id": "mov19",
            "title": "Second Chances",
            "description": "After a bitter divorce, a single father moves to a small coastal town to start over. There he meets a local bookstore owner who has her own painful past. Together, they slowly learn to trust and love again while helping each other heal. A warm, authentic story about resilience and new beginnings.",
            "genres": ["romance", "drama"],
            "year": 2023,
            "rating": 7.8,
        },

        # Adventure / Fantasy
        {
            "id": "mov5",
            "title": "Ocean Dreams",
            "description": "A passionate marine biologist on a research expedition discovers an ancient underwater civilization hidden in the deepest trenches of the ocean. As she learns about their advanced culture and technology, she must protect them from a corporation that wants to exploit their resources. An epic underwater adventure with environmental themes.",
            "genres": ["adventure", "fantasy"],
            "year": 2023,
            "rating": 8.0,
        },
        {
            "id": "mov8",
            "title": "Mountain Escape",
            "description": "A family embarks on what should be a simple hiking trip in the wilderness, but when they take a wrong turn, it becomes a desperate fight for survival. Facing harsh weather, dangerous wildlife, and dwindling supplies, they must overcome their differences and work together. A gripping survival story about family bonds tested by nature.",
            "genres": ["adventure", "drama"],
            "year": 2020,
            "rating": 7.6,
        },
        {
            "id": "mov20",
            "title": "The Lost Kingdom",
            "description": "An archaeologist discovers a map leading to a legendary lost kingdom hidden deep in the Amazon rainforest. Teaming up with a local guide, they navigate treacherous terrain and ancient traps while racing against treasure hunters. An adventure epic filled with mystery, danger, and archaeological wonders.",
            "genres": ["adventure", "action"],
            "year": 2022,
            "rating": 7.9,
        },
        {
            "id": "mov21",
            "title": "Dragon's Gate",
            "description": "In a mystical realm where dragons once ruled, a young blacksmith discovers she's the last descendant of dragon riders. She must embark on a perilous quest to find the legendary Dragon's Gate and restore balance between humans and the few remaining dragons. A fantasy adventure about heritage, destiny, and coexistence.",
            "genres": ["fantasy", "adventure"],
            "year": 2023,
            "rating": 8.2,
        },

        # Comedy
        {
            "id": "mov10",
            "title": "The Comedy Club",
            "description": "Five aspiring comedians with vastly different styles compete at a legendary comedy club for a chance at stardom and a spot on a national tour. Through failures and small victories, they learn about timing, authenticity, and the courage it takes to make people laugh. A hilarious yet touching look at the stand-up comedy world.",
            "genres": ["comedy", "drama"],
            "year": 2021,
            "rating": 7.7,
        },
        {
            "id": "mov22",
            "title": "Startup Chaos",
            "description": "Three college friends launch a tech startup from their garage, leading to absurd situations as they navigate investors, competitors, and their own incompetence. Despite constant disasters and hilarious mishaps, their friendship and determination keep them going. A comedy about entrepreneurship, failure, and never giving up.",
            "genres": ["comedy", "drama"],
            "year": 2022,
            "rating": 7.5,
        },
        {
            "id": "mov23",
            "title": "Wedding Crashers Anonymous",
            "description": "Two professional wedding crashers who live for free food and champagne accidentally crash a mob boss's daughter's wedding. What follows is a comedy of errors as they try to escape without being discovered, all while one of them genuinely falls for the bride's sister. Romantic comedy with slapstick humor.",
            "genres": ["comedy", "romance"],
            "year": 2021,
            "rating": 7.6,
        },

        # Horror / Psychological
        {
            "id": "mov24",
            "title": "The Haunting",
            "description": "A family moves into a Victorian mansion with a dark history, only to discover they're not alone. Supernatural occurrences escalate from unsettling to terrifying as they uncover the house's connection to a series of mysterious disappearances. A classic haunted house story with psychological depth and genuine scares.",
            "genres": ["horror", "thriller"],
            "year": 2022,
            "rating": 7.4,
        },
        {
            "id": "mov25",
            "title": "Fractured Mind",
            "description": "A psychiatrist treating a patient with dissociative identity disorder begins experiencing blackouts and memory loss herself. As the boundaries between doctor and patient blur, she questions what's real and whether she's treating the patient or becoming one. A psychological thriller about identity and sanity.",
            "genres": ["thriller", "horror"],
            "year": 2023,
            "rating": 7.8,
        },

        # Historical / War
        {
            "id": "mov26",
            "title": "Brothers in Arms",
            "description": "Based on true events, this film follows a company of soldiers during a pivotal World War II battle. Through the chaos of combat, it explores themes of brotherhood, sacrifice, and the cost of war. Unflinching in its portrayal of both the horror and heroism of battle, it honors those who served.",
            "genres": ["war", "drama"],
            "year": 2021,
            "rating": 8.4,
        },
        {
            "id": "mov27",
            "title": "The Resistance",
            "description": "Set during World War II, a group of ordinary citizens forms an underground resistance movement against occupation. Risking everything, they engage in espionage, sabotage, and rescue operations. A powerful story of courage, moral choices, and the strength of the human spirit in the darkest times.",
            "genres": ["war", "thriller"],
            "year": 2022,
            "rating": 8.2,
        },

        # Biographical / Inspirational
        {
            "id": "mov28",
            "title": "Against All Odds",
            "description": "The inspiring true story of an athlete who overcomes a devastating injury that doctors said would end her career. Through determination, innovative training, and unwavering support from her coach, she fights her way back to compete at the highest level. A testament to human resilience and the power of never giving up.",
            "genres": ["biography", "drama"],
            "year": 2023,
            "rating": 8.1,
        },
        {
            "id": "mov29",
            "title": "The Maestro",
            "description": "Following a deaf composer who revolutionized classical music despite his disability, this biopic explores his creative genius, personal struggles, and the masterpieces he created while battling his own demons. A moving portrait of artistic passion and the price of brilliance.",
            "genres": ["biography", "drama"],
            "year": 2022,
            "rating": 8.5,
        },

        # Social Issues / Contemporary
        {
            "id": "mov30",
            "title": "Voices Unheard",
            "description": "A documentary-style drama following three individuals from different backgrounds as they navigate systemic challenges in modern society. Their stories intersect in unexpected ways, highlighting issues of inequality, justice, and human connection. A thought-provoking examination of contemporary social issues with authentic performances.",
            "genres": ["drama", "social"],
            "year": 2023,
            "rating": 8.3,
        },

        # Additional Adventure movies (added to fix data sparsity for adventure fans)
        {
            "id": "mov31",
            "title": "Jungle Expedition",
            "description": "A team of explorers ventures into the uncharted depths of the Amazon rainforest to document rare species. When they discover evidence of a lost civilization, they must survive deadly wildlife, treacherous terrain, and rival treasure hunters who will stop at nothing. An action-packed survival adventure with breathtaking jungle cinematography.",
            "genres": ["adventure", "action"],
            "year": 2022,
            "rating": 7.8,
        },
        {
            "id": "mov32",
            "title": "Sahara Crossing",
            "description": "A group of adventurers attempts to cross the Sahara Desert to find an ancient oasis mentioned in medieval texts. Battling sandstorms, dehydration, and mysterious desert nomads, they uncover secrets of a vanished kingdom buried beneath the sands. A visually stunning desert adventure about perseverance and discovery.",
            "genres": ["adventure", "mystery"],
            "year": 2021,
            "rating": 7.7,
        },
        {
            "id": "mov33",
            "title": "Peak Challenge",
            "description": "An experienced mountaineer leads a diverse expedition to summit one of the world's most dangerous peaks. As they face avalanches, altitude sickness, and their own fears, the team must learn to trust each other completely. A gripping high-altitude adventure about teamwork, determination, and the limits of human endurance.",
            "genres": ["adventure", "drama"],
            "year": 2023,
            "rating": 8.0,
        },

        # Additional Horror movies (added to fix data sparsity for horror fans)
        {
            "id": "mov34",
            "title": "The Summoning",
            "description": "A group of paranormal investigators explores an abandoned asylum where patients mysteriously vanished decades ago. As night falls, they realize the spirits haven't left and are desperate to possess the living. A terrifying supernatural horror with genuine scares and atmospheric dread.",
            "genres": ["horror", "thriller"],
            "year": 2022,
            "rating": 7.6,
        },
        {
            "id": "mov35",
            "title": "Don't Look Back",
            "description": "After witnessing a gruesome accident, a woman is haunted by visions of a dark figure following her everywhere. As reality blurs with nightmare, she must uncover the truth before losing her sanity completely. A psychological horror exploring grief, guilt, and paranoia.",
            "genres": ["horror", "thriller"],
            "year": 2023,
            "rating": 7.9,
        },
        {
            "id": "mov36",
            "title": "The Infection",
            "description": "A mysterious virus transforms people into violent, mindless creatures. A small group of survivors barricades themselves in a medical facility, racing to find a cure while the infected close in. A intense zombie horror with social commentary on pandemic response and human nature.",
            "genres": ["horror", "thriller"],
            "year": 2021,
            "rating": 7.7,
        },

        # Additional War movies (added to fix data sparsity for war/historical fans)
        {
            "id": "mov37",
            "title": "The Last Battalion",
            "description": "A small unit of soldiers is trapped behind enemy lines during the final days of World War II. With dwindling supplies and overwhelming odds, they must complete a critical mission to prevent a massacre. A gritty, realistic war film about sacrifice, brotherhood, and the horrors of combat.",
            "genres": ["war", "drama"],
            "year": 2022,
            "rating": 8.1,
        },
        {
            "id": "mov38",
            "title": "Code Breakers",
            "description": "During World War II, a team of brilliant mathematicians and linguists work in secret to crack enemy codes that could turn the tide of war. Based on true events, this gripping drama reveals the unsung heroes who fought with their minds. A compelling war thriller about intelligence warfare.",
            "genres": ["war", "thriller"],
            "year": 2023,
            "rating": 8.3,
        },
        {
            "id": "mov39",
            "title": "No Man's Land",
            "description": "Set in the trenches of World War I, soldiers from opposing sides are trapped in no man's land after a failed assault. As they await rescue, they discover their shared humanity transcends nationalism. A powerful anti-war film about the futility and tragedy of conflict.",
            "genres": ["war", "drama"],
            "year": 2021,
            "rating": 8.4,
        },
    ]
    return movies


def simulate_user_history() -> Dict[str, List[str]]:
    """Simulate user viewing history with diverse preferences."""
    return {
        "user1": ["mov1", "mov9", "mov4", "mov13"],  # Sci-fi enthusiast
        "user2": ["mov2", "mov6", "mov10", "mov18"],  # Romance/drama lover
        "user3": ["mov3", "mov7", "mov11", "mov15"],  # Thriller/crime fan
        "user4": ["mov1", "mov5", "mov8", "mov20"],  # Adventure seeker
        "user5": ["mov26", "mov27", "mov29"],  # Historical/war films fan
        "user6": ["mov24", "mov25", "mov9"],  # Psychological thriller enthusiast
    }


def get_user_profile_embedding(
    client: Client,
    table_name: str,
    user_movies: List[str],
    records_map: Dict[str, Any],
    use_genre_weighting: bool = True
) -> List[float]:
    """Create user profile by averaging embeddings of watched movies.

    Args:
        use_genre_weighting: If True, weight movies by genre dominance.
                            Movies matching dominant genres get higher weights.
    """
    if not user_movies:
        return None

    # Collect movie data
    movies_data = []
    for movie_id in user_movies:
        if movie_id in records_map:
            payload = json.loads(records_map[movie_id]["payload_json"])
            movies_data.append({
                "vector": records_map[movie_id]["vector"],
                "genres": payload.get("genres", "").split(",")
            })

    if not movies_data:
        return None

    # Genre weighting logic
    if use_genre_weighting and len(movies_data) >= 2:
        # Count genre frequency
        genre_counter = Counter()
        for movie in movies_data:
            for genre in movie["genres"]:
                if genre:
                    genre_counter[genre.strip()] += 1

        # Calculate genre dominance (top genres that appear in >50% of movies)
        total_movies = len(movies_data)
        dominant_genres = {
            genre for genre, count in genre_counter.items()
            if count / total_movies >= 0.5
        }

        # Weight vectors
        weighted_vectors = []
        for movie in movies_data:
            movie_genres = {g.strip() for g in movie["genres"] if g}

            # Calculate weight based on genre overlap with dominant genres
            if dominant_genres:
                overlap = len(movie_genres & dominant_genres)
                total_dominant = len(dominant_genres)
                # Weight: 1.0 to 5.0 based on genre alignment (highly aggressive)
                weight = 1.0 + 4.0 * (overlap / total_dominant)
            else:
                weight = 1.0

            # Apply weight to vector
            weighted_vector = [v * weight for v in movie["vector"]]
            weighted_vectors.append(weighted_vector)

        # Average weighted vectors
        avg_vector = [
            sum(v[i] for v in weighted_vectors) / len(weighted_vectors)
            for i in range(len(weighted_vectors[0]))
        ]
    else:
        # Simple averaging (original behavior)
        vectors = [movie["vector"] for movie in movies_data]
        avg_vector = [sum(x) / len(vectors) for x in zip(*vectors)]

    return avg_vector


def rerank_by_genre_match(
    recommendations: List[Dict],
    user_movies: List[str],
    records_map: Dict[str, Any],
    genre_boost: float = 0.3,
    min_genre_ratio: float = 0.6
) -> List[Dict]:
    """Rerank recommendations to prioritize genre matches with balanced selection.

    This function implements a two-phase approach:
    1. Boost scores for genre-matching recommendations
    2. Ensure at least min_genre_ratio of top recommendations match dominant genres

    Args:
        recommendations: List of search results with 'id' and 'score'
        user_movies: List of movie IDs the user has watched
        records_map: Map of movie_id to record data
        genre_boost: Score boost for genre-matching recommendations
        min_genre_ratio: Minimum ratio of genre-matching recommendations (default: 0.6)

    Returns:
        Reranked list of recommendations with genre balance
    """
    if not recommendations:
        return recommendations

    # Calculate user's dominant genres
    genre_counter = Counter()
    for movie_id in user_movies:
        if movie_id in records_map:
            payload = json.loads(records_map[movie_id]["payload_json"])
            genres = payload.get("genres", "").split(",")
            for genre in genres:
                if genre:
                    genre_counter[genre.strip()] += 1

    # Identify dominant genres (>50% frequency)
    total_watched = len(user_movies)
    dominant_genres = {
        genre for genre, count in genre_counter.items()
        if count / total_watched >= 0.5
    }

    if not dominant_genres:
        return recommendations  # No reranking needed

    # Phase 1: Classify and boost recommendations
    genre_matches = []
    genre_mismatches = []

    for rec in recommendations:
        if rec["id"] in records_map:
            payload = json.loads(records_map[rec["id"]]["payload_json"])
            rec_genres = {g.strip() for g in payload.get("genres", "").split(",") if g}

            # Check if recommendation matches dominant genres
            if rec_genres & dominant_genres:
                # Apply boost and add to matches
                boosted_rec = {
                    **rec,
                    "adjusted_score": rec["score"] * (1.0 + genre_boost)
                }
                genre_matches.append(boosted_rec)
            else:
                # No boost for mismatches
                genre_mismatches.append({
                    **rec,
                    "adjusted_score": rec["score"]
                })

    # Phase 2: Balanced selection
    # Sort both lists by adjusted score
    genre_matches.sort(key=lambda x: x["adjusted_score"], reverse=True)
    genre_mismatches.sort(key=lambda x: x["adjusted_score"], reverse=True)

    # Calculate target counts
    total_desired = len(recommendations)
    min_genre_matches = int(np.ceil(total_desired * min_genre_ratio))
    max_non_matches = total_desired - min_genre_matches

    # Build final recommendation list
    final_recs = []

    # Add genre matches up to minimum required or available
    final_recs.extend(genre_matches[:min_genre_matches])

    # Fill remaining slots with best non-matches
    remaining_slots = total_desired - len(final_recs)
    if remaining_slots > 0:
        final_recs.extend(genre_mismatches[:remaining_slots])

    # If we don't have enough genre matches, fill with best available
    if len(final_recs) < total_desired:
        # Add remaining genre matches first
        additional_matches = genre_matches[min_genre_matches:]
        final_recs.extend(additional_matches[:total_desired - len(final_recs)])

        # Then add remaining mismatches if still needed
        if len(final_recs) < total_desired:
            additional_mismatches = genre_mismatches[max_non_matches:]
            final_recs.extend(additional_mismatches[:total_desired - len(final_recs)])

    # Sort final list by adjusted score
    final_recs.sort(key=lambda x: x["adjusted_score"], reverse=True)

    # Restore original score field for display
    for rec in final_recs:
        rec["score"] = rec.get("original_score", rec["score"])
        if "adjusted_score" in rec:
            del rec["adjusted_score"]
        if "original_score" in rec:
            del rec["original_score"]

    return final_recs


def main() -> None:
    """Run recommendation system examples."""
    print("=" * 80)
    print("DXMT Recommendation System Example")
    print("=" * 80)

    # Check dependencies
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        print("\n❌ Error: sentence-transformers not installed")
        print("   Install with: pip install sentence-transformers")
        return

    # Load embedding model
    print("\nLoading embedding model (all-MiniLM-L6-v2)...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embedding_dim = model.get_sentence_embedding_dimension()
    print(f"✅ Model loaded (dimension: {embedding_dim})")

    # Create client
    print("\nConnecting to DXMT server...")
    client = Client(host="localhost", port=9999)

    try:
        # Create table
        print("\nCreating movies table...")
        try:
            client.delete_table("movies")
        except Exception:
            pass

        client.create_table(
            name="movies",
            dimension=embedding_dim,
            distance_metric="cosine",
            index_type="hnsw",
        )
        print("✅ Table created")

        # Load movies
        print("\nLoading movie catalog...")
        movies = create_movie_catalog()

        # Generate embeddings from descriptions
        descriptions = [movie["description"] for movie in movies]
        embeddings = model.encode(descriptions, show_progress_bar=False, convert_to_numpy=True)

        # Prepare records
        records = []
        for movie, embedding in zip(movies, embeddings):
            records.append({
                "id": movie["id"],
                "vector": embedding.tolist(),
                "payload": {
                    "title": movie["title"],
                    "description": movie["description"],
                    "genres": ",".join(movie["genres"]),
                    "year": movie["year"],
                    "rating": movie["rating"],
                },
            })

        # Insert records
        print("\nInserting movies into DXMT...")
        client.insert_records(table_name="movies", records=records)
        print(f"✅ Inserted {len(records)} movies")

        # Get all records
        all_records = client.get_records(table_name="movies", ids=[m["id"] for m in movies])
        records_map = {record["id"]: record for record in all_records}

        # ================================================================
        # Example 1: Content-Based Recommendations (Similar Movies)
        # ================================================================
        print("\n" + "=" * 80)
        print("Example 1: Content-Based Recommendations")
        print("=" * 80)

        # User watched a movie, recommend similar ones
        watched_movie_id = "mov1"  # The Space Odyssey
        watched_movie = records_map[watched_movie_id]
        watched_payload = json.loads(watched_movie["payload_json"])

        print(f"\nUser watched: {watched_payload['title']}")
        print(f"Description: {watched_payload['description']}")
        print(f"Genres: {watched_payload['genres']}")

        # Find similar movies
        results = client.search(
            table_name="movies",
            vector=watched_movie["vector"],
            k=6,  # Top 6 (including itself)
        )

        print(f"\n{'─' * 80}")
        print("Recommended similar movies:")
        print(f"{'─' * 80}")

        for i, result in enumerate(results, 1):
            if result["id"] == watched_movie_id:
                continue  # Skip the movie itself

            record = records_map[result["id"]]
            payload = json.loads(record["payload_json"])

            print(f"\n{i - 1}. {payload['title']} (Similarity: {result['score']:.4f})")
            print(f"   {payload['description']}")
            print(f"   Genres: {payload['genres']} | Rating: {payload['rating']}/10")

        # ================================================================
        # Example 2: User Profile-Based Recommendations
        # ================================================================
        print("\n" + "=" * 80)
        print("Example 2: User Profile-Based Recommendations")
        print("=" * 80)

        user_history = simulate_user_history()

        for user_id, watched_movies in user_history.items():
            print(f"\n{user_id}'s viewing history:")
            for movie_id in watched_movies:
                if movie_id in records_map:
                    payload = json.loads(records_map[movie_id]["payload_json"])
                    print(f"  - {payload['title']} ({payload['genres']})")

            # Create user profile with genre weighting
            user_profile = get_user_profile_embedding(
                client, "movies", watched_movies, records_map,
                use_genre_weighting=True  # Enable genre-weighted recommendations
            )

            if user_profile:
                # Find recommendations
                # Fetch more results to ensure we have enough genre-matching options
                desired_recommendations = 5
                search_k = desired_recommendations + len(watched_movies) + 10  # Extra margin for genre filtering

                results = client.search(
                    table_name="movies",
                    vector=user_profile,
                    k=min(search_k, 30),  # Cap at catalog size
                )

                # Rerank to prioritize genre matches with stronger boost
                results = rerank_by_genre_match(
                    results, watched_movies, records_map, genre_boost=0.7
                )

                print(f"\nRecommendations for {user_id}:")
                rec_count = 0
                for result in results:
                    # Skip already watched
                    if result["id"] in watched_movies:
                        continue

                    rec_count += 1
                    if rec_count > desired_recommendations:
                        break

                    record = records_map[result["id"]]
                    payload = json.loads(record["payload_json"])
                    print(f"  {rec_count}. {payload['title']} (Score: {result['score']:.4f})")
                    print(f"     {payload['genres']} | Rating: {payload['rating']}/10")

                # Show statistics
                if rec_count < desired_recommendations:
                    print(f"  ℹ️  Only {rec_count} recommendations available (user watched {len(watched_movies)} movies)")

        # ================================================================
        # Example 3: Filtered Recommendations
        # ================================================================
        print("\n" + "=" * 80)
        print("Example 3: Filtered Recommendations (Recent High-Rated)")
        print("=" * 80)

        # User wants sci-fi movies from 2022 or later with rating >= 8.0
        user_preference_text = "space exploration and alien encounters"
        preference_embedding = model.encode(user_preference_text, convert_to_numpy=True)

        print(f"\nUser preference: \"{user_preference_text}\"")
        print("Filters: year >= 2022, rating >= 8.0")

        # Search with filters
        results = client.search(
            table_name="movies",
            vector=preference_embedding.tolist(),
            k=10,
            filter={
                "$and": [
                    {"year": {"$gte": 2022}},
                    {"rating": {"$gte": 8.0}},
                ]
            },
        )

        print(f"\n{'─' * 80}")
        print("Recommended movies (filtered):")
        print(f"{'─' * 80}")

        for i, result in enumerate(results, 1):
            record = records_map[result["id"]]
            payload = json.loads(record["payload_json"])

            print(f"\n{i}. {payload['title']} ({payload['year']})")
            print(f"   {payload['description']}")
            print(f"   Genres: {payload['genres']} | Rating: {payload['rating']}/10")
            print(f"   Relevance: {result['score']:.4f}")

        # ================================================================
        # Example 4: Genre-Based Recommendations
        # ================================================================
        print("\n" + "=" * 80)
        print("Example 4: Explore Genre (Thriller)")
        print("=" * 80)

        # Get a thriller movie as seed
        seed_movie_id = "mov3"  # Code Warriors (thriller)
        seed_movie = records_map[seed_movie_id]
        seed_payload = json.loads(seed_movie["payload_json"])

        print(f"\nSeed movie: {seed_payload['title']}")
        print(f"Genres: {seed_payload['genres']}")

        # Note: DXMT doesn't support filtering by array contains, so we'll do post-filtering
        results = client.search(
            table_name="movies",
            vector=seed_movie["vector"],
            k=12,
        )

        print(f"\n{'─' * 80}")
        print("More thrillers you might like:")
        print(f"{'─' * 80}")

        thriller_count = 0
        for result in results:
            if result["id"] == seed_movie_id:
                continue

            record = records_map[result["id"]]
            payload = json.loads(record["payload_json"])

            # Post-filter for thriller genre
            if "thriller" in payload["genres"].lower():
                thriller_count += 1
                print(f"\n{thriller_count}. {payload['title']} (Similarity: {result['score']:.4f})")
                print(f"   {payload['description']}")
                print(f"   Genres: {payload['genres']} | Rating: {payload['rating']}/10")

                if thriller_count >= 3:
                    break

        # ================================================================
        # Summary
        # ================================================================
        print("\n" + "=" * 80)
        print("Recommendation System Best Practices")
        print("=" * 80)
        print("""
Recommendation Strategies:

1. Content-Based Filtering:
   - Find items similar to what user liked
   - Use item descriptions/features as embeddings
   - Good for new users (cold start problem)

2. User Profile:
   - Create user embedding from interaction history
   - Average embeddings of liked/watched items
   - Find similar items to user profile

3. Hybrid Approach:
   - Combine content similarity with filters
   - Use metadata (genre, year, rating) for constraints
   - Balance relevance with diversity

4. Exploration:
   - Don't always show most similar items
   - Add random sampling for serendipity
   - Use lower similarity items for discovery

Implementation Tips:
  - Update user profile periodically
  - Cache user embeddings
  - Use filters to ensure relevance
  - Track and exclude already consumed items
  - Add recency bias for timely recommendations
  - A/B test different recommendation strategies

Performance:
  - Precompute user profiles offline
  - Use batch search for multiple users
  - Cache popular item recommendations
  - Use metadata filters to reduce search space

Real-World Applications:
  - E-commerce: Product recommendations
  - Streaming: Movies, music, podcasts
  - Social media: Content feed ranking
  - Job boards: Job-candidate matching
  - Dating apps: Profile matching
  - News: Article recommendations
        """)

    finally:
        # Keep table for validation tests
        print("\n✅ Keeping table 'movies' for validation tests...")
        # Commented out to keep table for test_recommendation_improvements.py
        # print("\nCleaning up...")
        # try:
        #     client.delete_table("movies")
        #     print("✅ Deleted table 'movies'")
        # except Exception as e:
        #     print(f"⚠️  Cleanup failed: {e}")

        client.close()
        print("\n✅ Done!")


if __name__ == "__main__":
    main()
