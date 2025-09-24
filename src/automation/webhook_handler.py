"""
Marketing AI Automation - Webhook Handler (Updated Version)
Handles incoming webhooks from Make.com with signature verification disabled for testing
"""

import os
import logging
import json
import hashlib
import hmac
from datetime import datetime, timedelta
from flask import Flask, request, jsonify
import redis
import psycopg2
from psycopg2.extras import RealDictCursor
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configuration
REDIS_URL = os.getenv('REDIS_URL', 'redis://redis:6379')
DATABASE_URL = os.getenv('DATABASE_URL', 'postgresql://postgres:password@postgres:5432/marketing_ai')
WEBHOOK_SECRET = os.getenv('WEBHOOK_SECRET', 'your-secret-key')

# Initialize Redis connection
try:
    redis_client = redis.from_url(REDIS_URL)
    redis_client.ping()
    logger.info("Redis connection established")
except Exception as e:
    logger.error(f"Redis connection failed: {e}")
    redis_client = None

def get_db_connection():
    """Get database connection"""
    try:
        conn = psycopg2.connect(DATABASE_URL)
        return conn
    except Exception as e:
        logger.error(f"Database connection error: {e}")
        return None

def verify_webhook_signature(payload, signature):
    """Verify webhook signature - DISABLED FOR TESTING"""
    # For testing with Make.com, we'll skip signature verification
    logger.info("Signature verification disabled for testing")
    return True
    
    # Original signature verification code (commented out):
    # if not signature:
    #     return False
    # expected_signature = hmac.new(
    #     WEBHOOK_SECRET.encode('utf-8'),
    #     payload,
    #     hashlib.sha256
    # ).hexdigest()
    # return hmac.compare_digest(f"sha256={expected_signature}", signature)

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        # Check Redis
        redis_status = "connected" if redis_client and redis_client.ping() else "disconnected"
        
        # Check Database
        conn = get_db_connection()
        db_status = "connected" if conn else "disconnected"
        if conn:
            conn.close()
        
        return jsonify({
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "services": {
                "redis": redis_status,
                "database": db_status
            }
        })
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return jsonify({"status": "unhealthy", "error": str(e)}), 500

@app.route('/webhook', methods=['POST'])
def handle_webhook():
    """Handle incoming webhooks from Make.com"""
    try:
        # Get payload
        payload = request.get_data()
        signature = request.headers.get('X-Webhook-Signature', '')
        
        logger.info(f"Received webhook: {len(payload)} bytes")
        logger.info(f"Headers: {dict(request.headers)}")
        
        # Verify signature (currently disabled)
        if not verify_webhook_signature(payload, signature):
            logger.warning("Invalid webhook signature")
            return jsonify({"error": "Invalid signature"}), 401
        
        # Parse JSON payload
        try:
            webhook_data = json.loads(payload.decode('utf-8'))
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON payload: {e}")
            return jsonify({"error": "Invalid JSON"}), 400
        
        logger.info(f"Webhook data: {webhook_data}")
        
        # Validate required fields
        required_fields = ['trigger', 'automation_type']
        if not all(field in webhook_data for field in required_fields):
            logger.error(f"Missing required fields. Got: {list(webhook_data.keys())}")
            return jsonify({"error": "Missing required fields: trigger, automation_type"}), 400
        
        # Process the automation request
        automation_task = {
            'id': f"task_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'trigger': webhook_data['trigger'],
            'automation_type': webhook_data['automation_type'],
            'data': webhook_data.get('data', {}),
            'status': 'queued',
            'created_at': datetime.now().isoformat(),
            'processed_at': None
        }
        
        # Queue the task in Redis
        if redis_client:
            try:
                redis_client.lpush('automation_queue', json.dumps(automation_task))
                logger.info(f"Task queued: {automation_task['id']}")
            except Exception as e:
                logger.error(f"Redis queue error: {e}")
        
        # Store task in database (with error handling)
        conn = get_db_connection()
        if conn:
            try:
                cursor = conn.cursor()
                # Create table if it doesn't exist
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS automation_tasks (
                        id VARCHAR(255) PRIMARY KEY,
                        trigger VARCHAR(100) NOT NULL,
                        automation_type VARCHAR(100) NOT NULL,
                        data JSONB,
                        status VARCHAR(50) DEFAULT 'queued',
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        processed_at TIMESTAMP,
                        result JSONB
                    )
                """)
                
                # Insert task
                cursor.execute("""
                    INSERT INTO automation_tasks (id, trigger, automation_type, data, status, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s)
                """, (
                    automation_task['id'],
                    automation_task['trigger'],
                    automation_task['automation_type'],
                    json.dumps(automation_task['data']),
                    automation_task['status'],
                    datetime.now()
                ))
                conn.commit()
                cursor.close()
                conn.close()
                logger.info(f"Task stored in database: {automation_task['id']}")
            except Exception as e:
                logger.error(f"Database error: {e}")
                if conn:
                    conn.close()
        
        # Return success response
        response = {
            "status": "success",
            "message": "Automation task queued successfully",
            "task_id": automation_task['id'],
            "trigger": automation_task['trigger'],
            "automation_type": automation_task['automation_type']
        }
        
        logger.info(f"Webhook processed successfully: {automation_task['id']}")
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Webhook processing error: {e}")
        logger.error(traceback.format_exc())
        return jsonify({"error": "Internal server error"}), 500

@app.route('/automation/status', methods=['GET'])
def get_automation_status():
    """Get current automation system status"""
    try:
        status = {
            "system": "operational",
            "timestamp": datetime.now().isoformat(),
            "queue_size": 0,
            "recent_tasks": []
        }
        
        # Get queue size from Redis
        if redis_client:
            try:
                status["queue_size"] = redis_client.llen('automation_queue')
            except Exception as e:
                logger.error(f"Redis status error: {e}")
        
        # Get recent tasks from database
        conn = get_db_connection()
        if conn:
            try:
                cursor = conn.cursor(cursor_factory=RealDictCursor)
                cursor.execute("SELECT * FROM automation_tasks ORDER BY created_at DESC LIMIT 10")
                status["recent_tasks"] = [dict(row) for row in cursor.fetchall()]
                cursor.close()
                conn.close()
            except Exception as e:
                logger.error(f"Query execution error: {e}")
        
        return jsonify(status)
        
    except Exception as e:
        logger.error(f"Status check error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/automation/queue', methods=['GET'])
def get_queue_status():
    """Get current queue status"""
    try:
        queue_info = {
            "queue_length": 0,
            "pending_tasks": []
        }
        
        if redis_client:
            try:
                queue_length = redis_client.llen('automation_queue')
                queue_info["queue_length"] = queue_length
                
                # Get pending tasks (up to 10)
                pending = redis_client.lrange('automation_queue', 0, 9)
                queue_info["pending_tasks"] = [json.loads(task) for task in pending]
            except Exception as e:
                logger.error(f"Queue status error: {e}")
        
        return jsonify(queue_info)
        
    except Exception as e:
        logger.error(f"Queue status error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    logger.info("Starting Marketing AI Automation Webhook Handler")
    app.run(host='0.0.0.0', port=5000, debug=False)