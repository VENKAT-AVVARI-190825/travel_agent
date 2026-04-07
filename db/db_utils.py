import sqlite3
import json
import pandas as pd
from datetime import date, datetime
from typing import List, Optional, Dict, Any
from pathlib import Path

# Handle both relative and absolute imports
try:
    from api.datamodels import User, Trip, ChatHistory, TripPlanModel
except ImportError:
    try:
        from api.datamodels import User, Trip, ChatHistory, TripPlanModel
    except ImportError:
        from api.datamodels import User, Trip, ChatHistory, TripPlanModel

# Database path
DB_PATH = Path(__file__).parent / "travel_ai.sqlite"

def get_connection():
    """Get database connection with row factory and timeout"""
    conn = sqlite3.connect(DB_PATH, timeout=30.0)  # 30 second timeout
    conn.row_factory = sqlite3.Row  # return dict-like rows
    return conn

# -------------------------
# UI Helper Functions (moved from app.py)
# -------------------------

def get_all_users() -> List[str]:
    """Get all user names for UI dropdown"""
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM users ORDER BY name")
        users = [row[0] for row in cursor.fetchall()]
        conn.close()
        return users
    except Exception as e:
        print(f"Error getting users: {e}")
        return []

def get_user_id_by_name(user_name: str) -> Optional[int]:
    """Get user ID from name"""
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM users WHERE name = ?", (user_name,))
        result = cursor.fetchone()
        conn.close()
        return result[0] if result else None
    except Exception as e:
        print(f"Error getting user ID: {e}")
        return None

def get_user_name_by_id(user_id: int) -> str:
    """Get user name from ID"""
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM users WHERE id = ?", (user_id,))
        result = cursor.fetchone()
        conn.close()
        return result[0] if result else "Unknown"
    except Exception as e:
        print(f"Error getting user name: {e}")
        return "Unknown"

def get_trips_by_user_name(user_name: str) -> List[Dict]:
    """Get all trips for a user by name"""
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT t.id, t.title, t.phase 
            FROM trips t 
            JOIN users u ON u.id = t.user_id 
            WHERE u.name = ?
            ORDER BY t.created_at DESC
        """, (user_name,))
        trips = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return trips
    except Exception as e:
        print(f"Error getting trips: {e}")
        return []

def load_table_as_dataframe(table_name: str) -> pd.DataFrame:
    """Load table data as DataFrame for UI display"""
    try:
        conn = sqlite3.connect(DB_PATH)  # Use regular connection for pandas
        df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
        conn.close()
        # If table has a created_at column, parse it as datetime
        if 'created_at' in df.columns:
            df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
        return df
    except Exception as e:
        print(f"Error loading table {table_name}: {e}")
        return pd.DataFrame()

def get_recent_chat_history(limit: int = 5) -> List[Dict]:
    """Get recent chat history for debugging"""
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT role, content, phase, created_at 
            FROM chat_history 
            ORDER BY created_at DESC 
            LIMIT ?
        """, (limit,))
        chat = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return chat
    except Exception as e:
        print(f"Error getting recent chat: {e}")
        return []

# -------------------------
# User Context Helper
# -------------------------
def get_trip_context(user_id: int, trip_id: int, phase: str) -> dict:
    """
    Returns trip context for a given user, trip, and phase using the trip_summary view.
    """
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT * FROM trip_summary WHERE id=? AND user_name=(SELECT name FROM users WHERE id=?) AND phase=?",
        (trip_id, user_id, phase)
    )
    row = cur.fetchone()
    conn.close()
    return dict(row) if row else {}

# -------------------------
# Trips
# -------------------------
def start_new_trip(user_id: int, request: dict) -> Trip:
    """
    Create a new Trip based on user input.
    """
    trip = Trip(user_id=user_id, **request, trip_status="draft")
    trip_id = create_trip(trip)
    trip.id = trip_id
    return trip

# -------------------------
# Chat History
# -------------------------
def save_chat_message_service(msg: ChatHistory) -> ChatHistory:
    """
    Save a chat message and return it with ID.
    """
    cid = save_chat_message(msg)
    msg.id = cid
    return msg

def load_chat_history_service(trip_id: int) -> List[dict]:
    """
    Load chat history for a trip.
    Always return as list of dicts for UI compatibility.
    """
    return load_chat_history(trip_id)

# --------------------------
# Helper converters
# --------------------------
def _serialize_value(value):
    """Convert Python ? SQLite compatible"""
    if isinstance(value, (dict, list)):
        return json.dumps(value)
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (date, datetime)):
        return str(value)
    return value

def _deserialize_value(value, expected_type):
    """Convert SQLite ? Python type"""
    if value is None:
        return None
    if expected_type == bool:
        return bool(value)
    if expected_type in (dict, list):
        try:
            return json.loads(value)
        except Exception:
            return {}
    if expected_type == date:
        return datetime.fromisoformat(value).date()
    if expected_type == datetime:
        return datetime.fromisoformat(value)
    return value

# --------------------------
# USERS
# --------------------------
def create_user(user: User) -> int:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO users (name, email, profile, travel_preferences, travel_constraints)
        VALUES (?, ?, ?, ?, ?)
    """, (user.name, user.email, user.profile, user.travel_preferences, user.travel_constraints))
    conn.commit()
    uid = cur.lastrowid
    conn.close()
    return int(uid) if uid is not None else 0

def get_user_by_id(user_id: int) -> Optional[User]:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM users WHERE id=?", (user_id,))
    row = cur.fetchone()
    conn.close()
    if row:
        return User(**dict(row))
    return None

# --------------------------
# TRIPS
# --------------------------
def create_trip(trip: Trip) -> int:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO trips (user_id, phase, title, origin, destination, trip_startdate, trip_enddate,
                           accommodation_type, no_of_adults, no_of_children, budget, currency, trip_status,
                           purpose, travel_preferences, travel_constraints)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        trip.user_id,
        trip.phase,
        trip.title,
        trip.origin,
        trip.destination,
        _serialize_value(trip.trip_startdate),
        _serialize_value(trip.trip_enddate),
        trip.accommodation_type,
        trip.no_of_adults,
        trip.no_of_children,
        trip.budget,
        trip.currency,
        trip.trip_status,
        trip.purpose,
        trip.travel_preferences,
        trip.travel_constraints
    ))
    conn.commit()
    tid = cur.lastrowid
    conn.close()
    return int(tid) if tid is not None else 0

def get_trip_by_id(trip_id: int) -> Optional[Trip]:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM trips WHERE id=?", (trip_id,))
    row = cur.fetchone()
    conn.close()
    if row:
        return Trip(**dict(row))
    return None

def update_trip_status(trip_id: int, trip_status: str):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("UPDATE trips SET trip_status=? WHERE id=?", (trip_status, trip_id))
    conn.commit()
    conn.close()

def update_trip_details(trip_id: int, **kwargs):
    """Update trip details with provided fields"""
    if not kwargs:
        return
    
    conn = get_connection()
    cur = conn.cursor()
    
    # Build dynamic update query
    updates = []
    params = []
    
    for field, value in kwargs.items():
        updates.append(f"{field}=?")
        params.append(_serialize_value(value))
    
    params.append(trip_id)
    query = f"UPDATE trips SET {', '.join(updates)} WHERE id=?"
    cur.execute(query, params)
    conn.commit()
    conn.close()

# --------------------------
# CHAT HISTORY
# --------------------------
def save_chat_message(msg: ChatHistory) -> int:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO chat_history (trip_id, user_id, role, phase, content, metadata, sequence_number, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        msg.trip_id,
        msg.user_id,
        msg.role,
        msg.phase,
        msg.content,
        msg.metadata,
        msg.sequence_number,
        _serialize_value(msg.created_at)
    ))
    conn.commit()
    cid = cur.lastrowid
    conn.close()
    return int(cid) if cid is not None else 0

def load_chat_history(trip_id: int) -> list[dict]:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT role, content FROM chat_history WHERE trip_id=? ORDER BY created_at", (trip_id,))
    rows = cur.fetchall()
    conn.close()
    # Always return as dicts for UI compatibility
    return [{"role": r[0], "content": r[1]} for r in rows]

# --------------------------
# TRIP PLANS
# --------------------------
def create_trip_plan(trip_plan: TripPlanModel) -> int:
    """Create a new trip plan in the database"""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO trip_plans (trip_id, itinerary_json, hotels_json, flights_json, daily_budget, total_estimated_cost, status, version, agent_metadata)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        trip_plan.trip_id,
        trip_plan.itinerary_json,
        trip_plan.hotels_json,
        trip_plan.flights_json,
        trip_plan.daily_budget,
        trip_plan.total_estimated_cost,
        trip_plan.status,
        trip_plan.version,
        trip_plan.agent_metadata
    ))
    conn.commit()
    plan_id = cur.lastrowid
    conn.close()
    return int(plan_id) if plan_id is not None else 0

def get_trip_plan_by_trip_id(trip_id: int, version: Optional[int] = None) -> Optional[TripPlanModel]:
    """Get trip plan by trip ID. If version not specified, gets latest version."""
    conn = get_connection()
    cur = conn.cursor()
    
    if version is not None:
        cur.execute("SELECT * FROM trip_plans WHERE trip_id=? AND version=?", (trip_id, version))
    else:
        cur.execute("""
            SELECT * FROM trip_plans 
            WHERE trip_id=? 
            ORDER BY version DESC 
            LIMIT 1
        """, (trip_id,))
    
    row = cur.fetchone()
    conn.close()
    
    if row:
        return TripPlanModel(**dict(row))
    return None

def get_all_trip_plan_versions(trip_id: int) -> List[TripPlanModel]:
    """Get all versions of trip plans for a trip"""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT * FROM trip_plans 
        WHERE trip_id=? 
        ORDER BY version DESC
    """, (trip_id,))
    rows = cur.fetchall()
    conn.close()
    
    return [TripPlanModel(**dict(row)) for row in rows]

def update_trip_plan_status(plan_id: int, status: str) -> bool:
    """Update trip plan status (draft, approved, rejected)"""
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("UPDATE trip_plans SET status=? WHERE id=?", (status, plan_id))
        conn.commit()
        updated = cur.rowcount > 0
        conn.close()
        return updated
    except Exception as e:
        print(f"Error updating trip plan status: {e}")
        return False

def delete_trip_plan(plan_id: int) -> bool:
    """Delete a trip plan"""
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("DELETE FROM trip_plans WHERE id=?", (plan_id,))
        conn.commit()
        deleted = cur.rowcount > 0
        conn.close()
        return deleted
    except Exception as e:
        print(f"Error deleting trip plan: {e}")
        return False

def save_travel_plan_to_db(travel_plan, trip_id: int, version: int = 1) -> int:
    """Helper function to save TravelPlan object to database"""
    trip_plan_model = TripPlanModel.from_travel_plan(travel_plan, trip_id, version)
    return create_trip_plan(trip_plan_model)

def get_trip_with_plan(trip_id: int) -> Optional[Dict]:
    """Get trip with its latest plan for UI display"""
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("""
            SELECT 
                t.*,
                tp.itinerary_json,
                tp.hotels_json,
                tp.flights_json,
                tp.daily_budget as plan_daily_budget,
                tp.total_estimated_cost,
                tp.status as plan_status,
                tp.generated_at as plan_generated_at,
                tp.version as plan_version
            FROM trips t
            LEFT JOIN trip_plans tp ON t.id = tp.trip_id 
                AND tp.version = (SELECT MAX(version) FROM trip_plans WHERE trip_id = t.id)
            WHERE t.id = ?
        """, (trip_id,))
        row = cur.fetchone()
        conn.close()
        
        return dict(row) if row else None
    except Exception as e:
        print(f"Error getting trip with plan: {e}")
        return None

# --------------------------
# ACTIONS LOGGING
# --------------------------
def log_action(trip_id: int, user_id: int, action_type: str, action_data: dict, phase: str = "phase3_autogen"):
    """Log agent actions and workflow steps"""
    try:
        conn = get_connection()
        cur = conn.cursor()
        
        # Check if actions table exists, create if not
        cur.execute("""
            CREATE TABLE IF NOT EXISTS actions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trip_id INTEGER,
                user_id INTEGER,
                phase TEXT,
                action_type TEXT,
                action_data TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (trip_id) REFERENCES trips(id),
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """)
        
        cur.execute("""
            INSERT INTO actions (trip_id, user_id, phase, action_type, action_data)
            VALUES (?, ?, ?, ?, ?)
        """, (trip_id, user_id, phase, action_type, json.dumps(action_data)))
        
        conn.commit()
        action_id = cur.lastrowid
        conn.close()
        return int(action_id) if action_id is not None else 0
    except Exception as e:
        print(f"Error logging action: {e}")
        return 0