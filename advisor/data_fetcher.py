import exa_py as exa
from datetime import datetime, timedelta
from langsmith import traceable

class ExaDataFetcher:
    def __init__(self, api_key):
        self.client = exa.Exa(api_key)
    
    @traceable(name="fetch_latest_news", run_type="retriever")
    def fetch_latest_news(self, days_back=3):
        """Fetch latest IPL news using Exa"""
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
        
        results = self.client.search_and_contents(
            "IPL cricket latest news updates fantasy league",
            text={"maxCharacters": 5000},
            num_results=10,
            start_published_date=start_date,
            end_published_date=end_date,
            type="auto"
        )
        
        processed_results = [
            {
                "title": result.title,
                "url": result.url,
                "text": result.text,
                "published_date": result.published_date
            }
            for result in results.results
        ]
        
        return processed_results
    
    @traceable(name="fetch_player_stats", run_type="retriever")
    def fetch_player_stats(self, player_name=None):
        """Fetch player statistics using Exa"""
        query = f"IPL cricket {player_name} statistics performance" if player_name else "IPL cricket player statistics performance"
        
        results = self.client.search_and_contents(
            query,
            text={"maxCharacters": 5000},
            num_results=5,
            type="auto",
            livecrawl="always"
        )
        
        processed_results = [
            {
                "title": result.title,
                "url": result.url,
                "text": result.text
            }
            for result in results.results
        ]
        
        return processed_results
    
    @traceable(name="fetch_injury_reports", run_type="retriever")
    def fetch_injury_reports(self):
        """Fetch injury reports using Exa"""
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
        
        results = self.client.search_and_contents(
            "IPL cricket player injuries updates team changes",
            text={"maxCharacters": 3000},
            num_results=5,
            start_published_date=start_date,
            end_published_date=end_date,
            type="auto",
            livecrawl="always"
        )
        
        processed_results = [
            {
                "title": result.title,
                "url": result.url,
                "text": result.text,
                "published_date": result.published_date
            }
            for result in results.results
        ]
        
        return processed_results
    
    @traceable(name="fetch_matchup_analysis", run_type="retriever")
    def fetch_matchup_analysis(self, team1=None, team2=None):
        """Fetch matchup analysis using Exa"""
        query = "IPL cricket matchup analysis team performance comparison"
        if team1 and team2:
            query = f"IPL cricket {team1} vs {team2} matchup analysis prediction"
        
        results = self.client.search_and_contents(
            query,
            text={"maxCharacters": 4000},
            num_results=3,
            type="auto",
            livecrawl="always" 
        )
        
        processed_results = [
            {
                "title": result.title,
                "url": result.url,
                "text": result.text
            }
            for result in results.results
        ]
        
        return processed_results