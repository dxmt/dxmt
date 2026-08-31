"""Test script to validate recommendation system improvements.

Tests:
1. Recommendation quantity (should get 5 recommendations per user)
2. Genre matching quality (sci-fi fans should get sci-fi recommendations)
3. Similarity score distribution
"""

import json
import sys
from collections import Counter
from typing import Dict, List, Any

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("❌ Error: sentence-transformers not installed")
    print("   Install with: pip install sentence-transformers")
    sys.exit(1)

from dxmt import Client


def analyze_recommendation_quality(
    user_id: str,
    watched_movies: List[str],
    recommendations: List[Dict],
    records_map: Dict[str, Any]
) -> Dict[str, Any]:
    """Analyze quality of recommendations for a user."""

    # Extract genres from watched movies
    watched_genres = Counter()
    for movie_id in watched_movies:
        if movie_id in records_map:
            payload = json.loads(records_map[movie_id]["payload_json"])
            genres = payload.get("genres", "").split(",")
            for genre in genres:
                if genre:
                    watched_genres[genre.strip()] += 1

    # Identify dominant genres (appear in >50% of watched movies)
    total_watched = len(watched_movies)
    dominant_genres = {
        genre for genre, count in watched_genres.items()
        if count / total_watched >= 0.5
    }

    # Analyze recommendations
    rec_genre_matches = 0
    rec_genres = []

    for rec in recommendations:
        if rec["id"] in records_map:
            payload = json.loads(records_map[rec["id"]]["payload_json"])
            genres = set(g.strip() for g in payload.get("genres", "").split(",") if g)
            rec_genres.extend(genres)

            # Check if recommendation matches dominant genres
            if dominant_genres and genres & dominant_genres:
                rec_genre_matches += 1

    # Calculate metrics
    genre_match_rate = (rec_genre_matches / len(recommendations) * 100) if recommendations else 0
    recommendation_count = len(recommendations)

    return {
        "user_id": user_id,
        "watched_count": total_watched,
        "dominant_genres": list(dominant_genres),
        "recommendation_count": recommendation_count,
        "genre_match_rate": genre_match_rate,
        "rec_genres": Counter(rec_genres).most_common(3),
        "scores": [rec["score"] for rec in recommendations]
    }


def test_recommendations():
    """Run comprehensive recommendation tests."""
    print("=" * 80)
    print("Recommendation System Improvement Validation")
    print("=" * 80)

    # Check dependencies
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        return

    # Load model
    print("\nLoading embedding model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print("✅ Model loaded")

    # Connect to server
    print("\nConnecting to DXMT server...")
    client = Client(host="localhost", port=9999)
    print("✅ Connected")

    try:
        # Get all records
        print("\nFetching movie catalog...")
        # Note: We assume the movies table exists from previous run
        # In production, we'd create it here
        try:
            # Try to get all records by searching with a dummy vector
            dummy_vector = [0.0] * 384
            all_results = client.search(
                table_name="movies",
                vector=dummy_vector,
                k=50  # Get all movies (increased to accommodate catalog expansion)
            )

            # Fetch full records
            movie_ids = [r["id"] for r in all_results]
            all_records = client.get_records(table_name="movies", ids=movie_ids)
            records_map = {record["id"]: record for record in all_records}

            print(f"✅ Loaded {len(records_map)} movies")

        except Exception as e:
            print(f"❌ Error: Could not fetch movies. Please run recommendation_system.py first.")
            print(f"   {e}")
            return

        # Test user histories
        user_histories = {
            "user1": ["mov1", "mov9", "mov4", "mov13"],  # Sci-fi enthusiast (4 sci-fi)
            "user2": ["mov2", "mov6", "mov10", "mov18"],  # Romance/drama
            "user3": ["mov3", "mov7", "mov11", "mov15"],  # Thriller/crime
            "user4": ["mov1", "mov5", "mov8", "mov20"],  # Adventure
            "user5": ["mov26", "mov27", "mov29"],  # Historical/war (3 movies)
            "user6": ["mov24", "mov25", "mov9"],  # Psychological thriller
        }

        print("\n" + "=" * 80)
        print("Test Results: Recommendation Quality")
        print("=" * 80)

        all_results = []

        for user_id, watched_movies in user_histories.items():
            # Get user profile embedding with genre weighting
            from recommendation_system import get_user_profile_embedding

            user_profile = get_user_profile_embedding(
                client, "movies", watched_movies, records_map,
                use_genre_weighting=True
            )

            if not user_profile:
                continue

            # Get recommendations
            desired_recommendations = 5
            search_k = desired_recommendations + len(watched_movies) + 2

            search_results = client.search(
                table_name="movies",
                vector=user_profile,
                k=min(search_k, 30)
            )

            # Filter out watched movies
            recommendations = [
                r for r in search_results
                if r["id"] not in watched_movies
            ][:desired_recommendations]

            # Analyze quality
            analysis = analyze_recommendation_quality(
                user_id, watched_movies, recommendations, records_map
            )
            all_results.append(analysis)

            # Print results
            print(f"\n{user_id}:")
            print(f"  Watched: {analysis['watched_count']} movies")
            print(f"  Dominant genres: {', '.join(analysis['dominant_genres']) if analysis['dominant_genres'] else 'None'}")
            print(f"  Recommendations: {analysis['recommendation_count']}")

            # Check if meets target
            if analysis['recommendation_count'] >= desired_recommendations:
                print(f"  ✅ Quantity: {analysis['recommendation_count']}/5 recommendations")
            else:
                print(f"  ⚠️  Quantity: Only {analysis['recommendation_count']}/5 recommendations")

            # Genre match quality
            if analysis['dominant_genres']:
                if analysis['genre_match_rate'] >= 60:
                    print(f"  ✅ Genre match: {analysis['genre_match_rate']:.1f}% (Good)")
                elif analysis['genre_match_rate'] >= 40:
                    print(f"  ⚠️  Genre match: {analysis['genre_match_rate']:.1f}% (Fair)")
                else:
                    print(f"  ❌ Genre match: {analysis['genre_match_rate']:.1f}% (Poor)")

            # Show recommended genres
            if analysis['rec_genres']:
                top_genres = ', '.join(f"{g}({c})" for g, c in analysis['rec_genres'])
                print(f"  Recommended genres: {top_genres}")

            # Similarity score distribution
            if analysis['scores']:
                scores = analysis['scores']
                if len(scores) >= 2:
                    score_drop = (scores[0] - scores[1]) / scores[0] * 100 if scores[0] > 0 else 0
                    if score_drop < 20:
                        print(f"  ✅ Score distribution: {scores[0]:.3f} → {scores[1]:.3f} (-{score_drop:.1f}%)")
                    else:
                        print(f"  ⚠️  Score distribution: {scores[0]:.3f} → {scores[1]:.3f} (-{score_drop:.1f}%)")

        # ================================================================
        # Summary Statistics
        # ================================================================
        print("\n" + "=" * 80)
        print("Summary Statistics")
        print("=" * 80)

        total_users = len(all_results)
        users_with_5_recs = sum(1 for r in all_results if r["recommendation_count"] >= 5)
        users_with_3_plus_recs = sum(1 for r in all_results if r["recommendation_count"] >= 3)

        avg_rec_count = sum(r["recommendation_count"] for r in all_results) / total_users
        avg_genre_match = sum(r["genre_match_rate"] for r in all_results if r["dominant_genres"]) / \
                         sum(1 for r in all_results if r["dominant_genres"])

        print(f"\nRecommendation Quantity:")
        print(f"  Users with 5 recommendations: {users_with_5_recs}/{total_users} ({users_with_5_recs/total_users*100:.1f}%)")
        print(f"  Users with 3+ recommendations: {users_with_3_plus_recs}/{total_users} ({users_with_3_plus_recs/total_users*100:.1f}%)")
        print(f"  Average recommendations per user: {avg_rec_count:.1f}")

        print(f"\nRecommendation Quality:")
        print(f"  Average genre match rate: {avg_genre_match:.1f}%")

        # Overall assessment
        print("\n" + "=" * 80)
        print("Overall Assessment")
        print("=" * 80)

        quantity_pass = users_with_5_recs >= total_users * 0.8  # 80% should have 5 recs
        quality_pass = avg_genre_match >= 60  # 60% genre match

        if quantity_pass and quality_pass:
            print("\n✅ ALL TESTS PASSED!")
            print(f"   Quantity: {users_with_5_recs}/{total_users} users get 5 recommendations")
            print(f"   Quality: {avg_genre_match:.1f}% average genre match rate")
        elif quantity_pass:
            print("\n⚠️  PARTIAL PASS - Quantity OK, Quality needs improvement")
            print(f"   Quantity: ✅ {users_with_5_recs}/{total_users} users get 5 recommendations")
            print(f"   Quality: ⚠️  {avg_genre_match:.1f}% genre match (target: 60%)")
        elif quality_pass:
            print("\n⚠️  PARTIAL PASS - Quality OK, Quantity needs improvement")
            print(f"   Quantity: ⚠️  Only {users_with_5_recs}/{total_users} users get 5 recommendations")
            print(f"   Quality: ✅ {avg_genre_match:.1f}% genre match")
        else:
            print("\n❌ TESTS FAILED - Both quantity and quality need improvement")
            print(f"   Quantity: ❌ Only {users_with_5_recs}/{total_users} users get 5 recommendations")
            print(f"   Quality: ❌ {avg_genre_match:.1f}% genre match (target: 60%)")

        # Detailed recommendations
        print("\n" + "=" * 80)
        print("Improvement Recommendations")
        print("=" * 80)

        if not quantity_pass:
            print("\n🔴 Recommendation Quantity:")
            print("   - Increase search_k parameter")
            print("   - Current: k = 5 + len(watched)")
            print("   - Suggested: k = 10 + len(watched)")

        if not quality_pass:
            print("\n🟡 Recommendation Quality:")
            print("   - Enable genre weighting in user profile")
            print("   - Consider hybrid scoring (semantic + genre)")
            print("   - Add post-filtering for dominant genres")

        # Show specific user issues
        problem_users = [r for r in all_results if
                        r["recommendation_count"] < 5 or
                        (r["dominant_genres"] and r["genre_match_rate"] < 60)]

        if problem_users:
            print("\n⚠️  Users needing attention:")
            for user in problem_users:
                issues = []
                if user["recommendation_count"] < 5:
                    issues.append(f"only {user['recommendation_count']} recs")
                if user["dominant_genres"] and user["genre_match_rate"] < 60:
                    issues.append(f"{user['genre_match_rate']:.0f}% genre match")

                print(f"   {user['user_id']}: {', '.join(issues)}")

    finally:
        client.close()
        print("\n✅ Test complete!")


if __name__ == "__main__":
    test_recommendations()
