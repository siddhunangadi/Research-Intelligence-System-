"""
FastAPI continuous evaluation router.
Triggers metric computation jobs and logs statistics.
"""

import os
import logging
from fastapi import APIRouter, HTTPException, BackgroundTasks
from evaluation.evaluation_runner import RISEvaluationRunner

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/evaluate", tags=["evaluation"])

@router.post("")
def trigger_evaluation(background_tasks: BackgroundTasks):
    """
    Spawns evaluation pipelines run inside background worker tasks.
    """
    dataset_path = os.getenv("GOLDEN_DATASET_PATH", "./evaluation/datasets/golden_dataset.json")
    
    try:
        runner = RISEvaluationRunner(golden_dataset_path=dataset_path)
        background_tasks.add_task(runner.run_evaluations)
        
        return {
            "message": "Evaluation pipeline started in background.",
            "golden_dataset": dataset_path,
            "results_path": "results.json"
        }
    except Exception as e:
        logger.error(f"Failed to launch evaluation pipeline task: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to trigger evaluation runner: {str(e)}")
