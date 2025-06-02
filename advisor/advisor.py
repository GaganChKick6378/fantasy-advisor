import json
from datetime import datetime, timedelta
from langsmith import traceable
import uuid

class FantasyIPLAdvisor:
    def __init__(self, data_fetcher, vector_store, llm):
        self.data_fetcher = data_fetcher
        self.vector_store = vector_store
        self.llm = llm
        self.last_refresh = None
    
    @traceable(name="refresh_static_data", run_type="chain")
    def refresh_static_data(self):
        """Refresh static data in vector store"""
        print("Refreshing static data...")
        
        # Fetch and store latest news
        news_items = self.data_fetcher.fetch_latest_news()
        for item in news_items:
            self.vector_store.add_document(
                f"NEWS: {item['title']}\nDate: {item['published_date']}\n{item['text']}",
                'news',
                item.get('published_date')
            )
        
        # Fetch and store injury reports
        injury_reports = self.data_fetcher.fetch_injury_reports()
        for item in injury_reports:
            self.vector_store.add_document(
                f"INJURY REPORT: {item['title']}\nDate: {item['published_date']}\n{item['text']}",
                'injury',
                item.get('published_date')
            )
        
        # Store general player stats
        player_stats = self.data_fetcher.fetch_player_stats()
        for item in player_stats:
            self.vector_store.add_document(
                f"PLAYER STATS: {item['title']}\n{item['text']}",
                'stats'
            )
        
        self.last_refresh = datetime.now()
        print("Static data refresh complete.")
        
        return {
            "news_count": len(news_items),
            "injury_reports_count": len(injury_reports),
            "player_stats_count": len(player_stats),
            "refresh_time": self.last_refresh.isoformat()
        }
    
    @traceable(name="get_advice", run_type="chain")
    def get_advice(self, query):
        """Get fantasy IPL advice based on user query"""
        session_id = str(uuid.uuid4())
        
        # Check if refresh is needed
        if not self.last_refresh or (datetime.now() - self.last_refresh) > timedelta(hours=12):
            refresh_result = self.refresh_static_data()
        
        query_lower = query.lower()
        dynamic_context = []
        
        # Extract player information
        player_context = self._extract_player_context(query, query_lower, dynamic_context)
        
        # Extract match information
        match_context = self._extract_match_context(query, query_lower, dynamic_context)
        
        # Get relevant context from vector store
        vector_results = self.vector_store.search(query)
        static_context = [item['text'] for item in vector_results]
        
        # Combine contexts
        all_context = dynamic_context + static_context
        context_text = "\n\n".join(all_context)
        
        # Generate response using LLM
        response_data = self.llm.generate_response(query, context_text)
        
        # Calculate confidence score based on available context
        confidence_score = self._calculate_confidence_score(
            query, 
            len(dynamic_context), 
            len(static_context),
            response_data.get('response', '')
        )
        
        return {
            "response": response_data.get('response', ''),
            "confidence_score": confidence_score,
            "context_sources": {
                "dynamic_context_count": len(dynamic_context),
                "static_context_count": len(static_context),
                "total_context_length": len(context_text)
            },
            "session_id": session_id,
            "query_type": self._classify_query_type(query_lower)
        }
    
    @traceable(name="extract_player_context", run_type="tool")
    def _extract_player_context(self, query, query_lower, dynamic_context):
        """Extract player-related context"""
        if any(term in query_lower for term in ['player', 'batsman', 'bowler', 'all-rounder', 'all rounder']):
            player_name = None
            for word in query.split():
                if word[0].isupper() and len(word) > 3 and word.lower() not in ['what', 'when', 'where', 'which', 'who', 'why', 'how']:
                    player_name = word
                    break
            
            player_stats = self.data_fetcher.fetch_player_stats(player_name)
            for item in player_stats:
                dynamic_context.append(f"LIVE PLAYER STATS: {item['title']}\n{item['text']}")
            
            return {
                "player_name": player_name,
                "stats_found": len(player_stats)
            }
        return {"player_name": None, "stats_found": 0}
    
    @traceable(name="extract_match_context", run_type="tool")
    def _extract_match_context(self, query, query_lower, dynamic_context):
        """Extract match-related context"""
        if any(term in query_lower for term in ['match', 'versus', 'vs', 'against', 'playing']):
            ipl_teams = ['Mumbai Indians', 'Chennai Super Kings', 'Royal Challengers Bangalore', 
                         'Kolkata Knight Riders', 'Delhi Capitals', 'Punjab Kings', 
                         'Rajasthan Royals', 'Sunrisers Hyderabad', 'Gujarat Titans', 'Lucknow Super Giants']
            
            team1 = None
            team2 = None
            
            for team in ipl_teams:
                if team.lower() in query_lower or team.split()[-1].lower() in query_lower:
                    if team1 is None:
                        team1 = team
                    elif team2 is None:
                        team2 = team
                        break
            
            matchup_analysis = self.data_fetcher.fetch_matchup_analysis(team1, team2)
            for item in matchup_analysis:
                dynamic_context.append(f"LIVE MATCHUP ANALYSIS: {item['title']}\n{item['text']}")
            
            return {
                "team1": team1,
                "team2": team2,
                "analysis_found": len(matchup_analysis)
            }
        return {"team1": None, "team2": None, "analysis_found": 0}
    
    def _calculate_confidence_score(self, query, dynamic_count, static_count, response):
        """Calculate confidence score based on available context and response quality"""
        base_score = 0.5
        
        # Boost score based on context availability
        if dynamic_count > 0:
            base_score += 0.2
        if static_count > 0:
            base_score += 0.15
        
        # Boost score based on response length and specificity
        if len(response) > 200:
            base_score += 0.1
        
        # Check for specific IPL terms in response
        ipl_terms = ['ipl', 'cricket', 'fantasy', 'player', 'team', 'match', 'runs', 'wickets']
        term_count = sum(1 for term in ipl_terms if term.lower() in response.lower())
        base_score += min(term_count * 0.02, 0.1)
        
        # Ensure score is between 0 and 1
        return min(max(base_score, 0.0), 1.0)
    
    def _classify_query_type(self, query_lower):
        """Classify the type of query for better tracking"""
        if any(term in query_lower for term in ['player', 'batsman', 'bowler']):
            return "player_query"
        elif any(term in query_lower for term in ['match', 'versus', 'vs']):
            return "match_query"
        elif any(term in query_lower for term in ['team', 'squad']):
            return "team_query"
        elif any(term in query_lower for term in ['strategy', 'captain', 'vice-captain']):
            return "strategy_query"
        else:
            return "general_query"