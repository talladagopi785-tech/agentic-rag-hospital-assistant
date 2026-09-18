from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any
import json, math, re, hashlib, uuid, os

app = FastAPI(
    title="Agentic RAG Hospital Assistant",
    description="Intelligent Hospital Discovery & Appointment Booking Platform powered by Agentic RAG",
    version="2.0.0"
)

BASE_DIR = Path(__file__).resolve().parent
HOSPITALS_FILE = BASE_DIR / "hospitals.json"
PATIENTS_FILE = BASE_DIR / "patients.json"
TEMPLATE_FILE = BASE_DIR / "templates" / "index.html"

def load_json(p: Path, default: Any):
    try:
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
        return default
    except Exception:
        return default

def save_json(p: Path, data: Any):
    p.write_text(json.dumps(data, indent=2), encoding="utf-8")

hospitals: List[Dict[str, Any]] = load_json(HOSPITALS_FILE, [])
patients: List[Dict[str, Any]] = load_json(PATIENTS_FILE, [])

# Vector embedding dimension
DIM = 64

def tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z0-9]+", str(text).lower())

def embed(text: str) -> List[float]:
    v = [0.0] * DIM
    tokens = tokenize(text)
    if not tokens:
        return v
    for t in tokens:
        idx = int(hashlib.sha256(t.encode("utf-8")).hexdigest(), 16) % DIM
        v[idx] += 1.0
    norm = math.sqrt(sum(x * x for x in v))
    return [x / norm for x in v] if norm > 0 else v

def cosine_similarity(a: List[float], b: List[float]) -> float:
    return sum(x * y for x, y in zip(a, b))

# Precompute document vectors
hospital_vectors = {
    h["id"]: embed(
        f"{h.get('name', '')} {h.get('city', '')} {h.get('area', '')} "
        f"{' '.join(h.get('specialties', []))} {' '.join(h.get('facilities', []))} "
        f"{h.get('description', '')}"
    )
    for h in hospitals
}

# Symptom and Specialty Knowledge Mapping for Agentic RAG
SYMPTOM_SPECIALTY_MAP = {
    "Cardiology": ["heart", "chest pain", "cardio", "cardiac", "angina", "palpitation", "bp", "hypertension", "vascular"],
    "Cardiac Surgery": ["bypass", "heart surgery", "valve", "cardiac surgery", "angioplasty", "open heart"],
    "Neurology": ["brain", "headache", "migraine", "stroke", "neuro", "seizure", "epilepsy", "nerve", "paralysis", "dizziness"],
    "Orthopedics": ["bone", "fracture", "joint", "knee", "back pain", "spine", "ortho", "orthopedic", "ligament", "arthritis", "hip"],
    "Pediatrics": ["child", "baby", "infant", "pediatric", "kid", "newborn", "vaccination", "pediatrician", "toddler"],
    "Dermatology": ["skin", "rash", "acne", "hair", "derma", "dermatologist", "eczema", "allergy", "scalp"],
    "Oncology": ["cancer", "tumor", "chemo", "radiation", "oncologist", "malignancy", "biopsy", "chemotherapy"],
    "Gynecology": ["women", "pregnancy", "maternity", "delivery", "gyno", "gynecologist", "period", "fertility", "obstetric", "prenatal"],
    "General Medicine": ["fever", "cough", "cold", "flu", "weakness", "infection", "consultation", "physician", "checkup", "fatigue"],
    "Emergency": ["emergency", "icu", "critical", "urgent", "ambulance", "trauma", "accident", "24 hours", "casualty"]
}

# TOOL 1: RAG Hospital Retrieval
def get_hospital_info(query: str) -> Dict[str, Any]:
    ql = query.lower()
    tokens = set(tokenize(ql))
    query_vec = embed(query)
    
    # 1. Detect target city
    all_cities = {h["city"].lower(): h["city"] for h in hospitals}
    target_city = next((c for c in all_cities if c in ql), None)
    
    # 2. Detect target specialties via symptoms or direct names
    target_specialties = set()
    for spec, keywords in SYMPTOM_SPECIALTY_MAP.items():
        if spec.lower() in ql:
            target_specialties.add(spec)
        elif any(kw in ql for kw in keywords):
            target_specialties.add(spec)
            
    # 3. Detect facility requests
    requires_emergency = any(k in ql for k in ["emergency", "24/7", "24 hours", "icu", "ambulance", "urgent"])

    # 4. Rank candidates with semantic similarity + agentic domain heuristics
    scored_hospitals = []
    for h in hospitals:
        hid = h["id"]
        h_vec = hospital_vectors.get(hid, [0.0] * DIM)
        sim_score = cosine_similarity(query_vec, h_vec)
        
        # Text matching bonuses
        text_rep = json.dumps(h).lower()
        token_overlap = sum(0.12 for w in tokens if w in text_rep)
        
        score = sim_score + token_overlap
        
        # Hard filtering / heavy weighting
        if target_city:
            if h["city"].lower() == target_city:
                score += 1.5
            else:
                score -= 1.0

        if target_specialties:
            h_specs = [s.lower() for s in h.get("specialties", [])]
            matches = [s for s in target_specialties if s.lower() in h_specs]
            if matches:
                score += 1.2 * len(matches)
            elif not target_city:
                score -= 0.5
                
        if requires_emergency and h.get("emergency") == "24 hours":
            score += 0.4
            
        # Rating boost (up to 0.5)
        score += (h.get("rating", 0.0) / 10.0)
        
        scored_hospitals.append((score, h))

    # Sort descending
    scored_hospitals.sort(reverse=True, key=lambda x: x[0])
    
    # Filter only positive candidates if filters applied
    filtered = [h for s, h in scored_hospitals if s > 0.1]
    top_results = filtered[:5] if filtered else [h for _, h in scored_hospitals[:5]]
    
    # Generate human-friendly reasoning message
    reasons = []
    if target_city:
        reasons.append(f"located in {all_cities[target_city]}")
    if target_specialties:
        reasons.append(f"specializing in {', '.join(sorted(target_specialties))}")
    if requires_emergency:
        reasons.append("offering 24/7 emergency care")
        
    reason_str = f" matching {', '.join(reasons)}" if reasons else ""
    msg = f"Found {len(top_results)} top-rated hospital(s){reason_str}."
    
    return {
        "success": True,
        "tool": "get_hospital_info",
        "message": msg,
        "hospitals": top_results,
        "criteria": {
            "city": all_cities.get(target_city) if target_city else None,
            "specialties": list(target_specialties),
            "emergency": requires_emergency
        }
    }

# TOOL 2: Appointment Management
def update_patient(
    action: str,
    patient_name: str = "Guest Patient",
    hospital_id: Any = None,
    doctor: str = "Chief Specialist",
    appointment_date: str = "",
    appointment_time: str = "",
    patient_id: Optional[str] = None
) -> Dict[str, Any]:
    global patients
    
    if action == "list":
        return {
            "success": True,
            "tool": "update_patient",
            "action": "list",
            "message": f"Retrieved {len(patients)} appointment(s).",
            "patients": patients
        }

    if action == "book":
        h = next((h for h in hospitals if str(h["id"]) == str(hospital_id)), None)
        if not h:
            return {"success": False, "tool": "update_patient", "message": "Hospital not found for booking."}

        today_str = datetime.now().strftime("%Y-%m-%d")
        apt_date = appointment_date.strip() or today_str
        apt_time = appointment_time.strip() or "10:00 AM"
        
        apt = {
            "appointment_id": f"APT-{uuid.uuid4().hex[:8].upper()}",
            "patient_name": patient_name.strip() or "Guest Patient",
            "hospital_id": h["id"],
            "hospital_name": h["name"],
            "city": h.get("city", ""),
            "area": h.get("area", ""),
            "doctor": doctor.strip() or "Chief Specialist",
            "appointment_date": apt_date,
            "appointment_time": apt_time,
            "status": "Confirmed",
            "created_at": datetime.now().isoformat()
        }
        patients.append(apt)
        save_json(PATIENTS_FILE, patients)
        return {
            "success": True,
            "tool": "update_patient",
            "action": "book",
            "message": f"Appointment {apt['appointment_id']} booked successfully at {h['name']} for {apt['patient_name']}.",
            "appointment": apt
        }

    if action == "cancel":
        target = None
        if patient_id:
            target = next((p for p in patients if p["appointment_id"].upper() == patient_id.strip().upper()), None)
        else:
            # Cancel most recent active appointment
            active = [p for p in patients if p.get("status") != "Cancelled"]
            if active:
                target = active[-1]

        if not target:
            return {
                "success": False,
                "tool": "update_patient",
                "action": "cancel",
                "message": "No active appointment found to cancel."
            }

        target["status"] = "Cancelled"
        target["cancelled_at"] = datetime.now().isoformat()
        save_json(PATIENTS_FILE, patients)
        return {
            "success": True,
            "tool": "update_patient",
            "action": "cancel",
            "message": f"Appointment {target['appointment_id']} at {target['hospital_name']} has been cancelled.",
            "appointment": target
        }

    return {"success": False, "message": f"Unknown action: {action}"}

# Intent Router (Agentic Decision Making)
def route(query: str) -> Dict[str, Any]:
    q = query.strip()
    ql = q.lower()
    
    # 1. Appointment Listing Intent
    list_keywords = ["my appointments", "my appointment", "show appointment", "show appointments", 
                     "list appointments", "list appointment", "view booking", "view bookings", 
                     "my bookings", "my booking", "check appointment", "check appointments"]
    if any(k in ql for k in list_keywords):
        return update_patient("list")

    # 2. Appointment Cancellation Intent
    cancel_keywords = ["cancel", "cancellation", "cancel appointment", "cancel my appointment", 
                       "cancel booking", "delete appointment", "revoke appointment"]
    # Check if query contains an appointment ID like APT-XXXXXX
    apt_id_match = re.search(r"APT-[A-Fa-f0-9]+", q, re.IGNORECASE)
    
    if apt_id_match or any(k in ql for k in cancel_keywords):
        target_id = apt_id_match.group(0).upper() if apt_id_match else None
        return update_patient("cancel", patient_id=target_id)

    # 3. Appointment Booking Intent
    book_keywords = ["book appointment", "book an appointment", "schedule appointment", 
                     "schedule an appointment", "book doctor", "book a doctor", "make appointment", 
                     "make an appointment", "reserve appointment", "reserve visit"]
    
    is_book_intent = any(k in ql for k in book_keywords) or (
        ("book" in ql or "schedule" in ql) and any(h["name"].lower() in ql or h["city"].lower() in ql for h in hospitals)
    )

    if is_book_intent:
        # Match target hospital
        matched_hospital = None
        for h in hospitals:
            if h["name"].lower() in ql:
                matched_hospital = h
                break
        if not matched_hospital:
            for h in hospitals:
                if h["city"].lower() in ql:
                    matched_hospital = h
                    break

        if not matched_hospital:
            return {
                "success": False,
                "tool": "route",
                "message": "Please specify the hospital name or city to book your appointment (e.g. 'Book appointment at Apollo Care in Hyderabad')."
            }

        # Extract patient name if given: e.g., "for Rahul" or "patient Rahul"
        patient_name = "Guest Patient"
        name_match = re.search(r"(?:for|patient|name)\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)?)", q)
        if name_match:
            patient_name = name_match.group(1).strip()

        # Extract doctor or specialty
        doctor = "Chief Specialist"
        for spec in matched_hospital.get("specialties", []):
            if spec.lower() in ql:
                doctor = f"Dr. ({spec} Specialist)"
                break

        return update_patient(
            action="book",
            patient_name=patient_name,
            hospital_id=matched_hospital["id"],
            doctor=doctor
        )

    # 4. Default to Agentic RAG Hospital Discovery
    return get_hospital_info(q)

# Request Models
class ChatRequest(BaseModel):
    message: str = Field(..., example="Find the best cardiology hospital in Hyderabad")

class BookRequest(BaseModel):
    hospital_id: Any
    patient_name: str = "Guest Patient"
    doctor: str = "General Specialist"
    appointment_date: str = ""
    appointment_time: str = ""

class CancelRequest(BaseModel):
    appointment_id: Optional[str] = None

# Routes
@app.get("/", response_class=HTMLResponse)
def home():
    if TEMPLATE_FILE.exists():
        return HTMLResponse(TEMPLATE_FILE.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>Agentic RAG Hospital Assistant</h1><p>Template file not found.</p>")

@app.post("/api/chat")
def chat_endpoint(req: ChatRequest):
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")
    return route(req.message)

@app.get("/api/hospitals")
def get_hospitals(
    city: Optional[str] = Query(None, description="Filter by city"),
    specialty: Optional[str] = Query(None, description="Filter by medical specialty"),
    search: Optional[str] = Query(None, description="Search term")
):
    results = hospitals
    if city:
        results = [h for h in results if h.get("city", "").lower() == city.lower()]
    if specialty:
        results = [h for h in results if any(s.lower() == specialty.lower() for s in h.get("specialties", []))]
    if search:
        s_term = search.lower()
        results = [h for h in results if s_term in json.dumps(h).lower()]
    return {"success": True, "count": len(results), "hospitals": results}

@app.get("/api/hospitals/{hospital_id}")
def get_hospital_detail(hospital_id: int):
    h = next((h for h in hospitals if h["id"] == hospital_id), None)
    if not h:
        raise HTTPException(status_code=404, detail="Hospital not found")
    return {"success": True, "hospital": h}

@app.get("/api/patients")
def get_appointments():
    return {"success": True, "count": len(patients), "patients": patients}

@app.post("/api/appointments/book")
def book_appointment(req: BookRequest):
    return update_patient(
        action="book",
        patient_name=req.patient_name,
        hospital_id=req.hospital_id,
        doctor=req.doctor,
        appointment_date=req.appointment_date,
        appointment_time=req.appointment_time
    )

@app.post("/api/appointments/cancel")
def cancel_appointment_endpoint(req: CancelRequest):
    return update_patient(
        action="cancel",
        patient_id=req.appointment_id
    )

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "agentic-rag-hospital-assistant",
        "hospitals_loaded": len(hospitals),
        "appointments_count": len(patients),
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)

