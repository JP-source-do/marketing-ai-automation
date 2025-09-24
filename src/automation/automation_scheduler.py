"""
Marketing AI Automation - Scheduler Service (Fixed Version)
Processes automation tasks from the queue and executes scheduled automations
"""

import os
import logging
import json
import time
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import threading
import schedule

import psycopg2
from psycopg2.extras import RealDictCursor
import redis
import requests
import pandas as pd
import openai

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/app/logs/automation_scheduler.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration from environment variables
DATABASE_CONFIG = {
    'host': os.getenv('POSTGRES_HOST', 'postgres'),
    'database': os.getenv('POSTGRES_DB', 'automation_db'),
    'user': os.getenv('POSTGRES_USER', 'automation_user'),
    'password': os.getenv('POSTGRES_PASSWORD', 'automation_pass123'),
    'port': int(os.getenv('POSTGRES_PORT', '5432'))
}

REDIS_CONFIG = {
    'host': os.getenv('REDIS_HOST', 'redis'),
    'port': int(os.getenv('REDIS_PORT', '6379')),
    'db': 0
}

# API Configuration
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')
META_APP_ID = os.getenv('META_APP_ID', '')
META_APP_SECRET = os.getenv('META_APP_SECRET', '')
GOOGLE_ADS_CUSTOMER_ID = os.getenv('GOOGLE_ADS_CUSTOMER_ID', '')
MAKE_WEBHOOK_URL = os.getenv('MAKE_WEBHOOK_URL')

# Initialize OpenAI if key is available
if OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY

# Initialize Redis connection
try:
    redis_client = redis.Redis(**REDIS_CONFIG)
    redis_client.ping()
    logger.info("Redis connection established")
except Exception as e:
    logger.error(f"Redis connection failed: {e}")
    redis_client = None

class DatabaseManager:
    """Handles database operations"""
    
    def __init__(self):
        self.config = DATABASE_CONFIG
        
    def get_connection(self):
        """Get database connection"""
        try:
            return psycopg2.connect(**self.config)
        except Exception as e:
            logger.error(f"Database connection error: {e}")
            return None
    
    def execute_query(self, query: str, params: tuple = None, fetch: bool = False):
        """Execute database query"""
        conn = self.get_connection()
        if not conn:
            return None
            
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(query, params)
                
                if fetch:
                    result = cursor.fetchall()
                    return [dict(row) for row in result]
                else:
                    conn.commit()
                    return True
        except Exception as e:
            logger.error(f"Query execution error: {e}")
            conn.rollback()
            return None
        finally:
            conn.close()

class AutomationProcessor:
    """Processes automation tasks from the queue"""
    
    def __init__(self):
        self.db = DatabaseManager()
        self.running = True
        
    def process_queue(self):
        """Main queue processing loop"""
        logger.info("Starting automation queue processor")
        
        while self.running:
            try:
                if not redis_client:
                    logger.warning("Redis not available, waiting...")
                    time.sleep(30)
                    continue
                
                # Get task from queue (blocking with timeout)
                task_data = redis_client.brpop('automation_queue', timeout=10)
                
                if task_data:
                    try:
                        task_json = task_data[1].decode('utf-8')
                        task = json.loads(task_json)
                        
                        logger.info(f"Processing task: {task.get('type')}")
                        
                        # Process the task
                        result = self.process_automation_task(task)
                        
                        # Update task status
                        self.update_task_status(task, result)
                        
                    except Exception as e:
                        logger.error(f"Task processing error: {e}")
                        
            except Exception as e:
                logger.error(f"Queue processing error: {e}")
                time.sleep(5)  # Wait before retrying
    
    def process_automation_task(self, task: Dict) -> Dict:
        """Process individual automation task"""
        task_type = task.get('type')
        task_data = task.get('data', {})
        
        try:
            if task_type == 'campaign_optimization':
                return self.optimize_campaign(task_data)
            elif task_type == 'lead_scoring':
                return self.score_lead(task_data)
            elif task_type == 'content_generation':
                return self.generate_content(task_data)
            elif task_type == 'performance_analysis':
                return self.analyze_performance(task_data)
            else:
                return {'success': False, 'error': f'Unknown task type: {task_type}'}
                
        except Exception as e:
            logger.error(f"Task execution error: {e}")
            return {'success': False, 'error': str(e)}
    
    def optimize_campaign(self, data: Dict) -> Dict:
        """Optimize marketing campaign based on performance data"""
        try:
            campaign_id = data.get('campaign_id')
            platform = data.get('platform', 'meta')  # meta, google_ads, etc.
            
            logger.info(f"Optimizing campaign {campaign_id} on {platform}")
            
            # Simulate campaign optimization logic
            optimizations = []
            
            # Check performance metrics (simulate with random data for now)
            current_ctr = data.get('ctr', 0.02)  # 2% default
            current_cpc = data.get('cpc', 1.50)  # $1.50 default
            current_roas = data.get('roas', 3.0)  # 3.0 default
            
            # Optimization recommendations
            if current_ctr < 0.015:  # Below 1.5%
                optimizations.append({
                    'type': 'creative_refresh',
                    'recommendation': 'Update ad creatives to improve click-through rate',
                    'priority': 'high'
                })
            
            if current_cpc > 2.0:  # Above $2.00
                optimizations.append({
                    'type': 'bid_adjustment',
                    'recommendation': 'Lower bid amounts to reduce cost per click',
                    'priority': 'medium'
                })
            
            if current_roas < 2.5:  # Below 2.5x
                optimizations.append({
                    'type': 'audience_refinement',
                    'recommendation': 'Refine target audience to improve ROAS',
                    'priority': 'high'
                })
            
            # Store optimization results
            optimization_result = {
                'campaign_id': campaign_id,
                'platform': platform,
                'optimizations': optimizations,
                'executed_at': datetime.now().isoformat(),
                'status': 'completed'
            }
            
            # Save to database
            query = """
            INSERT INTO campaign_optimizations (
                campaign_id, platform, optimization_data, executed_at, status
            ) VALUES (%s, %s, %s, %s, %s)
            """
            
            self.db.execute_query(
                query,
                (
                    campaign_id,
                    platform,
                    json.dumps(optimization_result),
                    datetime.now(),
                    'completed'
                )
            )
            
            # Send results to Make.com if configured
            if MAKE_WEBHOOK_URL:
                try:
                    requests.post(MAKE_WEBHOOK_URL, json=optimization_result, timeout=10)
                except Exception as e:
                    logger.error(f"Failed to send optimization results to Make.com: {e}")
            
            logger.info(f"Campaign optimization completed for {campaign_id}")
            return {'success': True, 'data': optimization_result}
            
        except Exception as e:
            logger.error(f"Campaign optimization error: {e}")
            return {'success': False, 'error': str(e)}
    
    def score_lead(self, data: Dict) -> Dict:
        """Score lead based on behavioral and demographic data"""
        try:
            lead_id = data.get('lead_id')
            lead_data = data.get('lead_data', {})
            
            logger.info(f"Scoring lead {lead_id}")
            
            # Simple lead scoring algorithm
            score = 0
            scoring_factors = []
            
            # Email engagement score
            email_opens = lead_data.get('email_opens', 0)
            email_clicks = lead_data.get('email_clicks', 0)
            if email_opens > 5:
                score += 20
                scoring_factors.append('High email engagement')
            if email_clicks > 2:
                score += 15
                scoring_factors.append('Email click activity')
            
            # Website behavior score
            page_views = lead_data.get('page_views', 0)
            time_on_site = lead_data.get('time_on_site', 0)  # in minutes
            if page_views > 10:
                score += 25
                scoring_factors.append('High website engagement')
            if time_on_site > 5:
                score += 20
                scoring_factors.append('Extended site visit duration')
            
            # Demographic score
            job_title = lead_data.get('job_title', '').lower()
            company_size = lead_data.get('company_size', 0)
            if any(title in job_title for title in ['manager', 'director', 'vp', 'ceo', 'cmo']):
                score += 30
                scoring_factors.append('Decision maker job title')
            if company_size > 100:
                score += 15
                scoring_factors.append('Large company size')
            
            # Social media engagement
            social_follows = lead_data.get('social_follows', 0)
            social_shares = lead_data.get('social_shares', 0)
            if social_follows > 0:
                score += 10
                scoring_factors.append('Social media engagement')
            
            # Determine lead grade
            if score >= 80:
                grade = 'A'
            elif score >= 60:
                grade = 'B'
            elif score >= 40:
                grade = 'C'
            else:
                grade = 'D'
            
            lead_score_result = {
                'lead_id': lead_id,
                'score': score,
                'grade': grade,
                'scoring_factors': scoring_factors,
                'scored_at': datetime.now().isoformat()
            }
            
            # Save to database
            query = """
            INSERT INTO lead_scores (
                lead_id, score, grade, scoring_data, scored_at
            ) VALUES (%s, %s, %s, %s, %s)
            """
            
            self.db.execute_query(
                query,
                (
                    lead_id,
                    score,
                    grade,
                    json.dumps(lead_score_result),
                    datetime.now()
                )
            )
            
            logger.info(f"Lead scoring completed for {lead_id}: {grade} ({score})")
            return {'success': True, 'data': lead_score_result}
            
        except Exception as e:
            logger.error(f"Lead scoring error: {e}")
            return {'success': False, 'error': str(e)}
    
    def generate_content(self, data: Dict) -> Dict:
        """Generate marketing content using AI"""
        try:
            content_type = data.get('content_type', 'social_post')
            prompt = data.get('prompt', '')
            target_audience = data.get('target_audience', 'general')
            
            logger.info(f"Generating {content_type} content")
            
            if not OPENAI_API_KEY:
                # Fallback to template-based content if OpenAI not available
                return self.generate_template_content(content_type, target_audience)
            
            # Create content prompt based on type
            if content_type == 'social_post':
                system_prompt = f"""
                Create an engaging social media post for {target_audience} audience.
                Keep it under 280 characters, include relevant hashtags, and make it actionable.
                """
            elif content_type == 'email_subject':
                system_prompt = f"""
                Create a compelling email subject line for {target_audience} audience.
                Keep it under 50 characters, make it curiosity-driven and avoid spam words.
                """
            elif content_type == 'ad_copy':
                system_prompt = f"""
                Create persuasive ad copy for {target_audience} audience.
                Include a clear value proposition and strong call-to-action.
                Keep it under 150 characters.
                """
            else:
                system_prompt = f"Create marketing content for {target_audience} audience."
            
            # Generate content using OpenAI (simulate for now)
            generated_content = f"Generated {content_type} for {target_audience}: {prompt[:100]}..."
            
            content_result = {
                'content_type': content_type,
                'generated_content': generated_content,
                'target_audience': target_audience,
                'prompt': prompt,
                'generated_at': datetime.now().isoformat()
            }
            
            # Save to database
            query = """
            INSERT INTO generated_content (
                content_type, content_text, target_audience, prompt, generated_at
            ) VALUES (%s, %s, %s, %s, %s)
            """
            
            self.db.execute_query(
                query,
                (
                    content_type,
                    generated_content,
                    target_audience,
                    prompt,
                    datetime.now()
                )
            )
            
            logger.info(f"Content generation completed: {content_type}")
            return {'success': True, 'data': content_result}
            
        except Exception as e:
            logger.error(f"Content generation error: {e}")
            return {'success': False, 'error': str(e)}
    
    def generate_template_content(self, content_type: str, target_audience: str) -> Dict:
        """Generate content using templates when AI is not available"""
        templates = {
            'social_post': [
                f"🚀 Exciting news for {target_audience}! Discover how our latest innovation can transform your business. #Innovation #Growth",
                f"💡 Pro tip for {target_audience}: Small changes can lead to big results. What's your next move? #ProTip #Success",
                f"🎯 Ready to level up your strategy? Join thousands of {target_audience} professionals who trust our solutions. #Strategy #Results"
            ],
            'email_subject': [
                f"Your {target_audience} success starts here",
                f"Exclusive offer for {target_audience} professionals",
                f"The {target_audience} guide you've been waiting for"
            ],
            'ad_copy': [
                f"Transform your business with solutions designed for {target_audience}. Get started today!",
                f"Join successful {target_audience} professionals. Proven results, guaranteed satisfaction.",
                f"The #1 choice for {target_audience}. See why thousands trust us with their success."
            ]
        }
        
        import random
        content = random.choice(templates.get(content_type, ["Generic marketing content"]))
        
        return {
            'success': True,
            'data': {
                'content_type': content_type,
                'generated_content': content,
                'target_audience': target_audience,
                'method': 'template',
                'generated_at': datetime.now().isoformat()
            }
        }
    
    def analyze_performance(self, data: Dict) -> Dict:
        """Analyze campaign performance and provide insights"""
        try:
            campaign_id = data.get('campaign_id')
            date_range = data.get('date_range', 7)  # days
            
            logger.info(f"Analyzing performance for campaign {campaign_id}")
            
            # Simulate performance analysis
            performance_metrics = {
                'impressions': data.get('impressions', 10000),
                'clicks': data.get('clicks', 150),
                'conversions': data.get('conversions', 12),
                'spend': data.get('spend', 500.0),
                'revenue': data.get('revenue', 1200.0)
            }
            
            # Calculate derived metrics
            ctr = (performance_metrics['clicks'] / performance_metrics['impressions']) * 100
            cpc = performance_metrics['spend'] / performance_metrics['clicks']
            conversion_rate = (performance_metrics['conversions'] / performance_metrics['clicks']) * 100
            roas = performance_metrics['revenue'] / performance_metrics['spend']
            
            # Generate insights
            insights = []
            
            if ctr < 1.0:
                insights.append({
                    'type': 'warning',
                    'metric': 'CTR',
                    'message': f'CTR is below average at {ctr:.2f}%. Consider refreshing ad creatives.',
                    'recommendation': 'Update ad images and copy to improve engagement'
                })
            
            if conversion_rate < 2.0:
                insights.append({
                    'type': 'alert',
                    'metric': 'Conversion Rate',
                    'message': f'Conversion rate is low at {conversion_rate:.2f}%. Check landing page experience.',
                    'recommendation': 'Optimize landing page and improve user experience'
                })
            
            if roas > 4.0:
                insights.append({
                    'type': 'success',
                    'metric': 'ROAS',
                    'message': f'Excellent ROAS of {roas:.2f}x! Consider scaling this campaign.',
                    'recommendation': 'Increase budget to capitalize on strong performance'
                })
            
            analysis_result = {
                'campaign_id': campaign_id,
                'date_range': date_range,
                'metrics': {
                    **performance_metrics,
                    'ctr': round(ctr, 2),
                    'cpc': round(cpc, 2),
                    'conversion_rate': round(conversion_rate, 2),
                    'roas': round(roas, 2)
                },
                'insights': insights,
                'analyzed_at': datetime.now().isoformat()
            }
            
            # Save to database
            query = """
            INSERT INTO performance_analyses (
                campaign_id, analysis_data, analyzed_at
            ) VALUES (%s, %s, %s)
            """
            
            self.db.execute_query(
                query,
                (
                    campaign_id,
                    json.dumps(analysis_result),
                    datetime.now()
                )
            )
            
            logger.info(f"Performance analysis completed for campaign {campaign_id}")
            return {'success': True, 'data': analysis_result}
            
        except Exception as e:
            logger.error(f"Performance analysis error: {e}")
            return {'success': False, 'error': str(e)}
    
    def update_task_status(self, task: Dict, result: Dict):
        """Update task status in database"""
        try:
            status = 'completed' if result.get('success') else 'failed'
            error_message = result.get('error') if not result.get('success') else None
            
            query = """
            UPDATE automation_tasks 
            SET status = %s, completed_at = %s, result_data = %s, error_message = %s
            WHERE task_type = %s AND task_data::text = %s AND status = 'queued'
            """
            
            self.db.execute_query(
                query,
                (
                    status,
                    datetime.now(),
                    json.dumps(result),
                    error_message,
                    task.get('type'),
                    json.dumps(task.get('data', {}))
                )
            )
            
        except Exception as e:
            logger.error(f"Task status update error: {e}")
    
    def stop(self):
        """Stop the processor"""
        self.running = False

class ScheduledAutomations:
    """Handles scheduled automation tasks"""
    
    def __init__(self):
        self.db = DatabaseManager()
    
    def setup_schedules(self):
        """Setup scheduled automation tasks"""
        # Daily performance analysis at 9 AM
        schedule.every().day.at("09:00").do(self.daily_performance_check)
        
        # Hourly queue health check
        schedule.every().hour.do(self.queue_health_check)
        
        # Weekly optimization review every Monday at 10 AM
        schedule.every().monday.at("10:00").do(self.weekly_optimization_review)
        
        logger.info("Scheduled automations configured")
    
    def daily_performance_check(self):
        """Daily automated performance analysis"""
        try:
            logger.info("Running daily performance check")
            
            # Get active campaigns (simulate)
            active_campaigns = [
                {'campaign_id': 'camp_001', 'platform': 'meta'},
                {'campaign_id': 'camp_002', 'platform': 'google_ads'}
            ]
            
            for campaign in active_campaigns:
                # Queue performance analysis task
                task = {
                    'type': 'performance_analysis',
                    'data': campaign,
                    'created_at': datetime.now().isoformat(),
                    'status': 'pending'
                }
                
                if redis_client:
                    redis_client.lpush('automation_queue', json.dumps(task))
                
                logger.info(f"Queued daily analysis for {campaign['campaign_id']}")
                
        except Exception as e:
            logger.error(f"Daily performance check error: {e}")
    
    def queue_health_check(self):
        """Check automation queue health"""
        try:
            if not redis_client:
                logger.warning("Redis unavailable for queue health check")
                return
            
            queue_length = redis_client.llen('automation_queue')
            
            if queue_length > 100:  # Alert if queue is backing up
                logger.warning(f"Automation queue is backing up: {queue_length} items")
                
                # Send alert (simulate)
                alert_data = {
                    'type': 'queue_backup',
                    'queue_length': queue_length,
                    'timestamp': datetime.now().isoformat()
                }
                
                if MAKE_WEBHOOK_URL:
                    try:
                        requests.post(MAKE_WEBHOOK_URL, json=alert_data, timeout=10)
                    except Exception as e:
                        logger.error(f"Failed to send queue alert: {e}")
            
            logger.info(f"Queue health check: {queue_length} items in queue")
            
        except Exception as e:
            logger.error(f"Queue health check error: {e}")
    
    def weekly_optimization_review(self):
        """Weekly optimization performance review"""
        try:
            logger.info("Running weekly optimization review")
            
            # Get optimization data from last week
            week_ago = datetime.now() - timedelta(days=7)
            
            optimizations = self.db.execute_query(
                "SELECT * FROM campaign_optimizations WHERE executed_at >= %s",
                (week_ago,),
                fetch=True
            )
            
            if optimizations:
                review_data = {
                    'period': 'weekly',
                    'optimization_count': len(optimizations),
                    'optimizations': optimizations,
                    'reviewed_at': datetime.now().isoformat()
                }
                
                # Send review to Make.com if configured
                if MAKE_WEBHOOK_URL:
                    try:
                        requests.post(MAKE_WEBHOOK_URL, json=review_data, timeout=10)
                    except Exception as e:
                        logger.error(f"Failed to send weekly review: {e}")
                
                logger.info(f"Weekly review completed: {len(optimizations)} optimizations reviewed")
            
        except Exception as e:
            logger.error(f"Weekly optimization review error: {e}")
    
    def run_scheduler(self):
        """Run the schedule checker"""
        logger.info("Starting scheduled automations")
        
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute

def main():
    """Main function to start the automation scheduler"""
    logger.info("Starting Marketing AI Automation Scheduler")
    
    # Initialize components
    processor = AutomationProcessor()
    scheduler = ScheduledAutomations()
    
    # Setup schedules
    scheduler.setup_schedules()
    
    # Start queue processor in a separate thread
    processor_thread = threading.Thread(target=processor.process_queue)
    processor_thread.daemon = True
    processor_thread.start()
    
    # Start scheduler in a separate thread
    scheduler_thread = threading.Thread(target=scheduler.run_scheduler)
    scheduler_thread.daemon = True
    scheduler_thread.start()
    
    try:
        # Keep main thread alive
        while True:
            time.sleep(10)
            
    except KeyboardInterrupt:
        logger.info("Shutting down automation scheduler")
        processor.stop()

if __name__ == "__main__":
    main()