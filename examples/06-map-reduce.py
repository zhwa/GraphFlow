"""
GraphFlow Example: Map-Reduce Processing
Ported from: PocketFlow cookbook/pocketflow-map-reduce

This example demonstrates parallel processing pattern where:
- Multiple items are processed independently (Map phase)
- Results are aggregated and analyzed (Reduce phase)
- Demonstrates batch processing capabilities
- Shows how to handle collections of data

Original PocketFlow pattern:
- ReadResumesNode, EvaluateResumesNode, ReduceResultsNode
- Sequential processing through shared state
- File-based data processing

GraphFlow adaptation:
- Functional map-reduce pattern
- Type-safe collection processing
- Flexible aggregation strategies
- Better error handling for batch operations
"""

import sys
import os
from typing import TypedDict, List, Dict, Any, Optional
import json

# Add the parent directory to Python path to import GraphFlow
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graphflow import StateGraph, Command, END, call_llm, configure_llm
from dataclasses import dataclass

# Data structures for resume processing
@dataclass
class Resume:
    filename: str
    candidate_name: str
    skills: List[str]
    experience_years: int
    education: str
    previous_roles: List[str]

@dataclass 
class Evaluation:
    candidate_name: str
    qualifies: bool
    score: float
    strengths: List[str]
    weaknesses: List[str]
    recommendation: str

# State schema for map-reduce processing
class MapReduceState(TypedDict):
    # Input data
    data_source: str                    # Path or identifier for data
    processing_criteria: Dict[str, Any] # Criteria for evaluation
    
    # Map phase
    raw_items: List[Dict[str, Any]]     # Raw data items
    processed_items: List[Dict[str, Any]] # Processed items
    processing_errors: List[str]        # Errors during processing
    
    # Reduce phase
    aggregated_results: Dict[str, Any]  # Final aggregated results
    summary_stats: Dict[str, float]     # Summary statistics
    recommendations: List[str]          # Final recommendations
    
    # Processing state
    total_items: int
    successful_items: int
    failed_items: int
    processing_complete: bool

def setup_llm_for_hiring(state: MapReduceState) -> Command:
    """Configure LLM provider for hiring evaluation."""
    print("🔧 Setting up LLM provider for hiring evaluation...")
    
    if os.environ.get("OPENAI_API_KEY"):
        configure_llm("openai", model="gpt-4")
        print("✅ Using OpenAI GPT-4 for resume evaluation")
    elif os.environ.get("ANTHROPIC_API_KEY"):
        configure_llm("anthropic", model="claude-3-sonnet-20240229")
        print("✅ Using Anthropic Claude for resume evaluation")
    else:
        try:
            configure_llm("ollama", model="llama2")
            call_llm("test")
            print("✅ Using Ollama (local) for resume evaluation")
        except Exception:
            print("❌ No LLM provider available - using fallback evaluation")
            print("\nTo use AI-powered evaluation, you need:")
            print("1. Set OPENAI_API_KEY environment variable, OR")
            print("2. Set ANTHROPIC_API_KEY environment variable, OR") 
            print("3. Start Ollama server: ollama serve")
    
    return Command(update={})

def load_data(state: MapReduceState) -> Command:
    """
    Load and prepare data for processing (equivalent to ReadResumesNode).
    """
    print("📁 Loading resume data...")
    
    # Simulate loading resume data (replace with actual file reading)
    sample_resumes = [
        {
            "filename": "john_doe.pdf",
            "candidate_name": "John Doe",
            "skills": ["Python", "Machine Learning", "SQL", "Docker"],
            "experience_years": 5,
            "education": "MS Computer Science",
            "previous_roles": ["Software Engineer", "Data Scientist"]
        },
        {
            "filename": "jane_smith.pdf", 
            "candidate_name": "Jane Smith",
            "skills": ["JavaScript", "React", "Node.js", "AWS"],
            "experience_years": 3,
            "education": "BS Software Engineering",
            "previous_roles": ["Frontend Developer", "Full Stack Developer"]
        },
        {
            "filename": "mike_johnson.pdf",
            "candidate_name": "Mike Johnson", 
            "skills": ["Java", "Spring", "Microservices", "Kubernetes"],
            "experience_years": 7,
            "education": "BS Computer Science",
            "previous_roles": ["Senior Developer", "Tech Lead", "Architect"]
        },
        {
            "filename": "sarah_wilson.pdf",
            "candidate_name": "Sarah Wilson",
            "skills": ["Python", "Data Science", "TensorFlow", "Statistics"],
            "experience_years": 4,
            "education": "PhD Data Science",
            "previous_roles": ["Data Analyst", "ML Engineer"]
        },
        {
            "filename": "david_brown.pdf",
            "candidate_name": "David Brown",
            "skills": ["C++", "System Design", "Linux", "Performance"],
            "experience_years": 8,
            "education": "MS Computer Engineering",
            "previous_roles": ["Systems Engineer", "Performance Engineer"]
        }
    ]
    
    print(f"✅ Loaded {len(sample_resumes)} resumes for processing")
    
    return Command(
        update={
            "raw_items": sample_resumes,
            "total_items": len(sample_resumes),
            "processing_errors": []
        }
    )

def map_process_items(state: MapReduceState) -> Command:
    """
    Process each item independently (Map phase - equivalent to EvaluateResumesNode).
    """
    print("🔄 Processing resumes (Map phase)...")
    
    raw_items = state["raw_items"]
    criteria = state["processing_criteria"]
    processed_items = []
    errors = []
    successful_count = 0
    
    for item in raw_items:
        try:
            # Create Resume object
            resume = Resume(
                filename=item["filename"],
                candidate_name=item["candidate_name"],
                skills=item["skills"],
                experience_years=item["experience_years"],
                education=item["education"],
                previous_roles=item["previous_roles"]
            )
            
            # Evaluate the resume
            evaluation = evaluate_resume(resume, criteria)
            
            # Convert to dict for state storage
            processed_item = {
                "filename": resume.filename,
                "candidate_name": resume.candidate_name,
                "qualifies": evaluation.qualifies,
                "score": evaluation.score,
                "strengths": evaluation.strengths,
                "weaknesses": evaluation.weaknesses,
                "recommendation": evaluation.recommendation
            }
            
            processed_items.append(processed_item)
            successful_count += 1
            
            print(f"✅ Processed {resume.candidate_name}: {'Qualified' if evaluation.qualifies else 'Not Qualified'} (Score: {evaluation.score:.1f})")
            
        except Exception as e:
            error_msg = f"Error processing {item.get('filename', 'unknown')}: {e}"
            errors.append(error_msg)
            print(f"❌ {error_msg}")
    
    return Command(
        update={
            "processed_items": processed_items,
            "processing_errors": state["processing_errors"] + errors,
            "successful_items": successful_count,
            "failed_items": len(errors)
        }
    )

def reduce_aggregate_results(state: MapReduceState) -> Command:
    """
    Aggregate and analyze all processed items (Reduce phase - equivalent to ReduceResultsNode).
    """
    print("📊 Aggregating results (Reduce phase)...")
    
    processed_items = state["processed_items"]
    
    if not processed_items:
        return {
            "aggregated_results": {"error": "No items to aggregate"},
            "summary_stats": {},
            "recommendations": ["No data available for analysis"],
            "processing_complete": True
        }
    
    # Aggregate statistics
    qualified_candidates = [item for item in processed_items if item["qualifies"]]
    total_candidates = len(processed_items)
    qualification_rate = len(qualified_candidates) / total_candidates if total_candidates > 0 else 0
    
    # Calculate score statistics
    scores = [item["score"] for item in processed_items]
    avg_score = sum(scores) / len(scores) if scores else 0
    max_score = max(scores) if scores else 0
    min_score = min(scores) if scores else 0
    
    # Analyze skills
    all_skills = []
    for item in processed_items:
        # Extract skills from the original data
        resume_data = next((r for r in state["raw_items"] if r["candidate_name"] == item["candidate_name"]), {})
        all_skills.extend(resume_data.get("skills", []))
    
    skill_counts = {}
    for skill in all_skills:
        skill_counts[skill] = skill_counts.get(skill, 0) + 1
    
    top_skills = sorted(skill_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    
    # Generate summary
    aggregated_results = {
        "total_candidates": total_candidates,
        "qualified_candidates": len(qualified_candidates),
        "qualification_rate": qualification_rate,
        "qualified_names": [c["candidate_name"] for c in qualified_candidates],
        "top_candidates": sorted(qualified_candidates, key=lambda x: x["score"], reverse=True)[:3],
        "common_skills": top_skills,
        "average_score": avg_score
    }
    
    summary_stats = {
        "total_processed": total_candidates,
        "success_rate": state["successful_items"] / state["total_items"] if state["total_items"] > 0 else 0,
        "qualification_rate": qualification_rate,
        "average_score": avg_score,
        "score_range": max_score - min_score,
        "top_score": max_score
    }
    
    # Generate recommendations
    recommendations = generate_hiring_recommendations(aggregated_results, summary_stats)
    
    # Print detailed results
    print(f"\n📈 Processing Summary:")
    print(f"   • Total candidates: {total_candidates}")
    print(f"   • Qualified candidates: {len(qualified_candidates)} ({qualification_rate:.1%})")
    print(f"   • Average score: {avg_score:.1f}")
    print(f"   • Success rate: {summary_stats['success_rate']:.1%}")
    
    print(f"\n🏆 Top Qualified Candidates:")
    for i, candidate in enumerate(aggregated_results["top_candidates"], 1):
        print(f"   {i}. {candidate['candidate_name']} (Score: {candidate['score']:.1f})")
    
    print(f"\n💡 Recommendations:")
    for i, rec in enumerate(recommendations, 1):
        print(f"   {i}. {rec}")
    
    return Command(
        update={
            "aggregated_results": aggregated_results,
            "summary_stats": summary_stats,
            "recommendations": recommendations,
            "processing_complete": True
        }
    )

def evaluate_resume(resume: Resume, criteria: Dict[str, Any]) -> Evaluation:
    """Evaluate a single resume against criteria using LLM for intelligent assessment."""
    
    # Create comprehensive resume context for LLM
    resume_text = f"""
Candidate: {resume.candidate_name}
Experience: {resume.experience_years} years
Education: {resume.education}
Skills: {', '.join(resume.skills)}
Previous Roles: {' → '.join(resume.previous_roles)}
    """.strip()
    
    # Create criteria context
    required_skills = criteria.get("required_skills", [])
    min_experience = criteria.get("min_experience_years", 0)
    min_score = criteria.get("min_score", 70)
    position_type = criteria.get("position_type", "Software Engineer")
    
    criteria_text = f"""
Position: {position_type}
Required Skills: {', '.join(required_skills)}
Minimum Experience: {min_experience} years
Minimum Score for Qualification: {min_score}/100
    """.strip()
    
    # Use LLM for intelligent evaluation
    evaluation_prompt = f"""You are an experienced technical recruiter evaluating a resume for a position. 

RESUME:
{resume_text}

JOB REQUIREMENTS:
{criteria_text}

Please evaluate this candidate and provide:
1. A score from 0-100 based on fit for the position
2. 3-5 key strengths (what makes them a good candidate)
3. 2-4 areas of concern or weaknesses 
4. Whether they qualify (score >= {min_score})
5. A hiring recommendation

Format your response as:
SCORE: [0-100]
QUALIFIES: [YES/NO]
STRENGTHS: [bullet list]
WEAKNESSES: [bullet list]
RECOMMENDATION: [one sentence recommendation]

Be thorough but concise in your evaluation."""

    try:
        llm_response = call_llm(evaluation_prompt)
        
        # Parse LLM response
        score, qualifies, strengths, weaknesses, recommendation = parse_evaluation_response(llm_response)
        
        print(f"🤖 LLM evaluated {resume.candidate_name}: {'Qualified' if qualifies else 'Not Qualified'} (Score: {score:.1f})")
        
    except Exception as e:
        print(f"⚠️  LLM evaluation failed for {resume.candidate_name}: {e}")
        # Fallback to rule-based evaluation
        score, qualifies, strengths, weaknesses, recommendation = evaluate_resume_fallback(resume, criteria)
        print(f"📊 Fallback evaluated {resume.candidate_name}: {'Qualified' if qualifies else 'Not Qualified'} (Score: {score:.1f})")
    
    return Evaluation(
        candidate_name=resume.candidate_name,
        qualifies=qualifies,
        score=score,
        strengths=strengths,
        weaknesses=weaknesses,
        recommendation=recommendation
    )

def parse_evaluation_response(response: str) -> tuple:
    """Parse LLM evaluation response into structured data."""
    score = 0.0
    qualifies = False
    strengths = []
    weaknesses = []
    recommendation = "No recommendation provided"
    
    lines = response.split('\n')
    current_section = None
    
    for line in lines:
        line = line.strip()
        if line.startswith("SCORE:"):
            try:
                score = float(line.replace("SCORE:", "").strip())
            except:
                score = 0.0
        elif line.startswith("QUALIFIES:"):
            qualifies = "YES" in line.upper()
        elif line.startswith("STRENGTHS:"):
            current_section = "strengths"
        elif line.startswith("WEAKNESSES:"):
            current_section = "weaknesses"
        elif line.startswith("RECOMMENDATION:"):
            recommendation = line.replace("RECOMMENDATION:", "").strip()
            current_section = None
        elif line.startswith("- ") or line.startswith("• "):
            bullet_text = line[2:].strip()
            if current_section == "strengths":
                strengths.append(bullet_text)
            elif current_section == "weaknesses":
                weaknesses.append(bullet_text)
    
    return score, qualifies, strengths, weaknesses, recommendation

def evaluate_resume_fallback(resume: Resume, criteria: Dict[str, Any]) -> tuple:
    """Fallback evaluation when LLM is unavailable."""
    score = 0.0
    strengths = []
    weaknesses = []
    
    # Required skills evaluation
    required_skills = criteria.get("required_skills", [])
    skill_matches = [skill for skill in required_skills if skill.lower() in [s.lower() for s in resume.skills]]
    skill_score = len(skill_matches) / len(required_skills) if required_skills else 1.0
    score += skill_score * 40  # 40% weight for skills
    
    if skill_matches:
        strengths.append(f"Has {len(skill_matches)}/{len(required_skills)} required skills: {', '.join(skill_matches)}")
    else:
        weaknesses.append(f"Missing required skills: {', '.join(required_skills)}")
    
    # Experience evaluation
    min_experience = criteria.get("min_experience_years", 0)
    if resume.experience_years >= min_experience:
        exp_score = min(resume.experience_years / (min_experience + 3), 1.0)  # Diminishing returns
        score += exp_score * 30  # 30% weight for experience
        strengths.append(f"{resume.experience_years} years experience (required: {min_experience})")
    else:
        weaknesses.append(f"Only {resume.experience_years} years experience (required: {min_experience})")
    
    # Education evaluation
    education_bonus = 0
    if "PhD" in resume.education:
        education_bonus = 15
        strengths.append("PhD degree")
    elif "MS" in resume.education or "Master" in resume.education:
        education_bonus = 10
        strengths.append("Master's degree")
    elif "BS" in resume.education or "Bachelor" in resume.education:
        education_bonus = 5
        strengths.append("Bachelor's degree")
    
    score += education_bonus
    
    # Role progression evaluation
    if len(resume.previous_roles) >= 2:
        score += 10  # Career progression bonus
        strengths.append(f"Career progression: {' → '.join(resume.previous_roles)}")
    
    # Additional skills bonus
    additional_skills = [skill for skill in resume.skills if skill.lower() not in [s.lower() for s in required_skills]]
    if additional_skills:
        score += min(len(additional_skills) * 2, 10)  # Up to 10 points for additional skills
        strengths.append(f"Additional valuable skills: {', '.join(additional_skills[:3])}")
    
    # Normalize score to 0-100
    score = min(score, 100)
    
    # Determine qualification
    min_score = criteria.get("min_score", 70)
    qualifies = score >= min_score
    
    # Generate recommendation
    if score >= 90:
        recommendation = "Excellent candidate - highly recommend for interview"
    elif score >= 80:
        recommendation = "Strong candidate - recommend for interview"
    elif score >= 70:
        recommendation = "Good candidate - consider for interview"
    elif score >= 60:
        recommendation = "Marginal candidate - review carefully"
    else:
        recommendation = "Does not meet requirements"
    
    return score, qualifies, strengths, weaknesses, recommendation

def generate_hiring_recommendations(results: Dict[str, Any], stats: Dict[str, float]) -> List[str]:
    """Generate hiring recommendations based on aggregated results using LLM."""
    
    # Prepare context for LLM
    context = f"""
HIRING ANALYSIS SUMMARY:
- Total Candidates: {results['total_candidates']}
- Qualified Candidates: {results['qualified_candidates']}
- Qualification Rate: {stats['qualification_rate']:.1%}
- Average Score: {stats['average_score']:.1f}/100
- Score Range: {stats.get('score_range', 0):.1f}
- Top Score: {stats.get('top_score', 0):.1f}

TOP CANDIDATES:
{chr(10).join([f"- {c['candidate_name']}: {c['score']:.1f}/100 - {c['recommendation']}" for c in results['top_candidates'][:3]])}

COMMON SKILLS IN POOL:
{chr(10).join([f"- {skill}: {count} candidates" for skill, count in results['common_skills'][:5]])}

QUALIFIED CANDIDATES:
{chr(10).join([f"- {name}" for name in results['qualified_names']])}
    """.strip()
    
    llm_prompt = f"""You are an experienced hiring manager reviewing a candidate pool analysis. Based on the data below, provide strategic hiring recommendations.

{context}

Please provide 5-7 specific, actionable recommendations for the hiring team. Focus on:
1. Overall pool quality assessment
2. Priority candidates to interview
3. Skills analysis insights
4. Process improvements
5. Next steps

Format as a numbered list of clear, actionable recommendations."""

    try:
        llm_response = call_llm(llm_prompt)
        
        # Parse numbered recommendations from LLM response
        recommendations = []
        lines = llm_response.split('\n')
        for line in lines:
            line = line.strip()
            # Look for numbered items (1., 2., etc. or 1), 2), etc.)
            if (line and (line[0].isdigit() or line.startswith('•') or line.startswith('-'))):
                # Clean up the line
                clean_line = line
                if line[0].isdigit():
                    # Remove numbering like "1." or "1)"
                    clean_line = line.split('.', 1)[-1].split(')', 1)[-1].strip()
                elif line.startswith('•') or line.startswith('-'):
                    clean_line = line[1:].strip()
                
                if clean_line:
                    recommendations.append(clean_line)
        
        if recommendations:
            print(f"🤖 Generated {len(recommendations)} LLM-powered hiring recommendations")
            return recommendations
        else:
            raise Exception("No recommendations parsed from LLM response")
            
    except Exception as e:
        print(f"⚠️  LLM recommendation generation failed: {e}")
        # Fallback to rule-based recommendations
        return generate_hiring_recommendations_fallback(results, stats)

def generate_hiring_recommendations_fallback(results: Dict[str, Any], stats: Dict[str, float]) -> List[str]:
    """Fallback recommendation generation when LLM is unavailable."""
    recommendations = []
    
    # Overall assessment
    if stats["qualification_rate"] >= 0.6:
        recommendations.append("Strong candidate pool - multiple qualified applicants available")
    elif stats["qualification_rate"] >= 0.3:
        recommendations.append("Moderate candidate pool - some qualified applicants available")
    else:
        recommendations.append("Weak candidate pool - consider expanding search criteria")
    
    # Top candidates
    if results["top_candidates"]:
        top_candidate = results["top_candidates"][0]
        recommendations.append(f"Priority interview: {top_candidate['candidate_name']} (score: {top_candidate['score']:.1f})")
    
    # Skill analysis
    if results["common_skills"]:
        top_skill = results["common_skills"][0]
        recommendations.append(f"Most common skill in pool: {top_skill[0]} ({top_skill[1]} candidates)")
    
    # Score-based recommendations
    if stats["average_score"] < 60:
        recommendations.append("Consider adjusting requirements - average scores are low")
    elif stats["average_score"] > 85:
        recommendations.append("High-quality candidate pool - can be selective in final decisions")
    
    print(f"📊 Generated {len(recommendations)} fallback hiring recommendations")
    return recommendations

def build_mapreduce_graph():
    """Build the map-reduce processing graph."""
    graph = StateGraph(MapReduceState)
    
    # Add processing nodes
    graph.add_node("setup_llm", setup_llm_for_hiring)
    graph.add_node("load_data", load_data)
    graph.add_node("map_process", map_process_items)
    graph.add_node("reduce_aggregate", reduce_aggregate_results)
    
    # Sequential flow
    graph.add_edge("setup_llm", "load_data")
    graph.add_edge("load_data", "map_process")
    graph.add_edge("map_process", "reduce_aggregate")
    graph.add_edge("reduce_aggregate", END)
    
    graph.set_entry_point("setup_llm")
    
    return graph.compile()

def main():
    """Main function to run map-reduce resume processing."""
    print("GraphFlow Map-Reduce Resume Processing Example")
    print("=" * 50)
    
    # Build the processing pipeline
    resume_processor = build_mapreduce_graph()
    
    # Define processing criteria
    criteria_sets = [
        {
            "job_title": "Senior Python Developer",
            "required_skills": ["Python", "SQL"],
            "min_experience_years": 3,
            "min_score": 70
        },
        {
            "job_title": "Frontend Developer", 
            "required_skills": ["JavaScript", "React"],
            "min_experience_years": 2,
            "min_score": 65
        },
        {
            "job_title": "Data Scientist",
            "required_skills": ["Python", "Machine Learning", "Statistics"],
            "min_experience_years": 4,
            "min_score": 75
        }
    ]
    
    for criteria in criteria_sets:
        print(f"\n{'='*60}")
        print(f"Processing for: {criteria['job_title']}")
        print(f"Required skills: {', '.join(criteria['required_skills'])}")
        print(f"Min experience: {criteria['min_experience_years']} years")
        print("=" * 60)
        
        # Initialize processing state
        initial_state = {
            "data_source": "resume_database",
            "processing_criteria": criteria,
            "raw_items": [],
            "processed_items": [],
            "processing_errors": [],
            "aggregated_results": {},
            "summary_stats": {},
            "recommendations": [],
            "total_items": 0,
            "successful_items": 0,
            "failed_items": 0,
            "processing_complete": False
        }
        
        try:
            # Run the map-reduce pipeline
            result = resume_processor.invoke(initial_state)
            
            print(f"\n✅ Processing complete!")
            print(f"Final statistics: {json.dumps(result['summary_stats'], indent=2)}")
            
        except Exception as e:
            print(f"❌ Error during processing: {e}")

def custom_criteria_mode():
    """Interactive mode for custom job criteria."""
    print("\nCustom Job Criteria Mode")
    print("Define your own job requirements!")
    print("-" * 30)
    
    # Get custom criteria from user
    job_title = input("Job title: ").strip()
    skills_input = input("Required skills (comma-separated): ").strip()
    required_skills = [skill.strip() for skill in skills_input.split(",") if skill.strip()]
    
    try:
        min_experience = int(input("Minimum years of experience: ").strip())
        min_score = float(input("Minimum score (0-100): ").strip())
    except ValueError:
        print("Invalid input. Using defaults.")
        min_experience = 3
        min_score = 70
    
    criteria = {
        "job_title": job_title or "Custom Position",
        "required_skills": required_skills,
        "min_experience_years": min_experience,
        "min_score": min_score
    }
    
    print(f"\nProcessing resumes for: {criteria['job_title']}")
    
    resume_processor = build_mapreduce_graph()
    
    initial_state = {
        "data_source": "resume_database",
        "processing_criteria": criteria,
        "raw_items": [],
        "processed_items": [],
        "processing_errors": [],
        "aggregated_results": {},
        "summary_stats": {},
        "recommendations": [],
        "total_items": 0,
        "successful_items": 0,
        "failed_items": 0,
        "processing_complete": False
    }
    
    try:
        result = resume_processor.invoke(initial_state)
        print(f"\n🎯 Processing complete for {criteria['job_title']}!")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--custom":
        custom_criteria_mode()
    else:
        main()

"""
GraphFlow Map-Reduce Benefits:

1. Parallel Processing Pattern:
   - Clear separation of Map and Reduce phases
   - Scalable to large datasets
   - Independent item processing

2. Type Safety:
   - Structured data with dataclasses
   - Type-safe state management
   - Clear input/output contracts

3. Error Handling:
   - Graceful failure handling
   - Error collection and reporting
   - Partial processing support

4. Comprehensive Analysis:
   - Detailed evaluation criteria
   - Statistical aggregation
   - Actionable recommendations

5. Flexibility:
   - Configurable processing criteria
   - Custom evaluation logic
   - Multiple job types support

Usage:
# Demo mode with predefined criteria
python 07-map-reduce.py

# Interactive mode with custom criteria
python 07-map-reduce.py --custom

To extend this example:
1. Add real file processing (PDF, Word documents)
2. Implement parallel processing with threading/asyncio
3. Add machine learning for resume scoring
4. Integrate with ATS (Applicant Tracking Systems)
5. Add support for different data sources (databases, APIs)
"""
