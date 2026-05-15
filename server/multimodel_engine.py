import os, json, asyncio, httpx, logging, re
from dotenv import load_dotenv

load_dotenv()
GEMINI_KEY = os.getenv("GEMINI_API_KEY")
logger = logging.getLogger("MultiModelEngine")

class MultiModelEngine:
    def __init__(self):
        self.orchestrator = "gemini-2.5-flash-lite"
        self.api_url = "http://127.0.0.1:3001/api/chat"
        self.available_models = [
            "llama-3.3-70b-versatile", # For reasoning/explanation
            "llama-3.1-8b-instant",    # For general tasks
            "codestral",               # For code
        ]

    async def _call_gemini(self, system_prompt: str, user_prompt: str) -> str:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.orchestrator}:generateContent?key={GEMINI_KEY}"
        payload = {
            "system_instruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
            "generationConfig": {"temperature": 0.1, "responseMimeType": "application/json"}
        }
        async with httpx.AsyncClient() as client:
            resp = await client.post(url, json=payload, timeout=30.0)
            if resp.status_code == 200:
                return resp.json()['candidates'][0]['content']['parts'][0]['text']
            else:
                raise Exception(f"Gemini Router Error: {resp.text}")

    async def deconstruct_prompt(self, prompt: str) -> list:
        sys_prompt = (
            "You are an expert AI Router. Your job is to break down the user's prompt into a set of distinct subtasks. "
            "Assign each subtask to the most appropriate model from the following list:\n"
            f"{self.available_models}\n\n"
            "- Use 'llama-3.3-70b-versatile' for theoretical explanation, reasoning, stats, and MUST BE USED for generating JSON configurations to draw charts/graphs.\n"
            "- Use 'codestral' exclusively for writing programming code. NEVER use codestral to generate charts/graphs.\n"
            "- Use 'llama-3.1-8b-instant' for simple questions.\n\n"
            "OUTPUT ONLY A JSON ARRAY OF OBJECTS in the exact format:\n"
            "[{\"task\": \"task description\", \"model\": \"model_id\"}]"
        )
        raw_json = await self._call_gemini(sys_prompt, prompt)
        try:
            tasks = json.loads(raw_json)
            if not isinstance(tasks, list): tasks = [tasks]
            return tasks
        except Exception as e:
            logger.error(f"Failed to parse Gemini routing JSON: {e}")
            # Fallback
            return [{"task": prompt, "model": "llama-3.3-70b-versatile"}]

    async def _execute_subtask(self, task_info: dict, history: str, domain: str) -> dict:
        task = task_info.get("task", "")
        model_id = task_info.get("model", "llama-3.3-70b-versatile")
        
        # We append the specific subtask to the history as the new prompt
        payload = {
            "message": f"Please complete this specific task: {task}",
            "model_id": model_id,
            "domain": domain,
            "history": history
        }
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(self.api_url, data=payload, timeout=60.0)
                if resp.status_code == 200:
                    return {"model": model_id, "task": task, "reply": resp.json().get("reply", "")}
                else:
                    return {"model": model_id, "task": task, "reply": f"Error {resp.status_code}"}
        except Exception as e:
            return {"model": model_id, "task": task, "reply": f"Failed: {str(e)}"}

    async def run_pipeline(self, prompt: str, history: str, domain: str) -> str:
        logger.info(f"MultiModel Pipeline Triggered for prompt: {prompt}")
        
        # 1. Break down the prompt
        tasks = await self.deconstruct_prompt(prompt)
        logger.info(f"Subtasks generated: {tasks}")
        
        # 2. Execute concurrently
        coroutines = [self._execute_subtask(t, history, domain) for t in tasks]
        results = await asyncio.gather(*coroutines)
        
        # 3. Synthesize the response (The Composer Layer)
        raw_outputs = "\n---\n".join([f"MODEL: {res['model']}\nTASK: {res['task']}\nOUTPUT:\n{res['reply']}" for res in results])
        
        synthesis_prompt = (
            "You are the Lead Analytical Architect. Your job is to take multiple specialist outputs and compose them into ONE unified, professional report.\n\n"
            "RULES:\n"
            "- Remove all duplicate explanations and repeated headers.\n"
            "- Unify the tone: professional, doctoral, and analytical.\n"
            "- Logical Order: ## Concept -> ## Formula -> [CHART] -> ## Interpretation -> ## Python Implementation.\n"
            "- Merge any fragmented Python code into one single, production-grade block.\n"
            "- CRITICAL: Preserve any <chart_params> tags exactly as they are. Move them to the very end of the response.\n"
            "- CRITICAL: Ensure LaTeX formulas are mathematically correct and use standard symbols (S_0, \\mu, \\sigma).\n"
            "- Add an '## Assumptions' block at the top if relevant (e.g. S0=100, \\mu=0.05, etc.).\n"
            "- Add a final '## Interpretation' section that explains what the results/simulation actually means in real-world terms.\n"
            "OUTPUT ONLY THE FINAL MARKDOWN REPORT."
        )
        
        final_report = await self._call_gemini(synthesis_prompt, f"ORIGINAL USER PROMPT: {prompt}\n\nSPECIALIST OUTPUTS:\n{raw_outputs}")
        
        # Extract chart params from the final report and ensure they are at the end
        all_chart_params = ""
        matches = re.findall(r'<chart_params>.*?</chart_params>', final_report, flags=re.DOTALL)
        for m in matches:
            all_chart_params += m + "\n"
            final_report = final_report.replace(m, "")
            
        return final_report.strip() + "\n\n" + all_chart_params.strip()
