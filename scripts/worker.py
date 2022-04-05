import time
import io
import ujson
import logging.config
from multiprocessing import Process, Queue
from dotenv import load_dotenv
load_dotenv()
from modules import intent
from jobqueue_worker import Job, Result, ResultStatus, basic_worker
from jobqueue_worker.config import loggers

logging.config.dictConfig(loggers.logger_config)

import logging

LOG = logging.getLogger(__name__.split(".")[0])

# one job success on job fails
job_success = False


def handler(job: Job, stream: io.BytesIO):
    try:
        #intents = intent.Intent(json.loads(stream))
        #result_data = intents.compute_labeled_scores()
        settings = ujson.loads(job.parameters)
        
        if "compare" not in settings:
            return Result(status=ResultStatus.FAILED)
        if settings["compare"] not in ["labeled", "unlabeled"]:
            return Result(status=ResultStatus.FAILED)
        labeled = settings["compare"] == "labeled"

        stream_read = stream.read().decode("utf-8")
        result_data = ujson.dumps(stream_read)
        data = ujson.loads(result_data)
    
        intents = intent.Intent(result_data)
        output = intents.compute_labeled_scores_fast()
        print(len(output))
        result_stream = io.StringIO(output)
        result_stream.write()
        result_stream.seek(0)

        #intents = intent.Intent()
        result = Result(
            status=ResultStatus.SUCCESS,
            params={
                "worker": "Job Worker",
                "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "input_data": settings,
            },
            blob_name=job.id,
            blob_data=result_stream)

        LOG.info("Handled job")
        return result

    except Exception as ex:
        LOG.exception("Function failed")
    
    LOG.info("Handled job as failed")
    return Result(status=ResultStatus.FAILED)

def run_worker():
    """Run basic_worker in a different process.
    """
    queue = Queue()
    p = Process(target=basic_worker, args=(handler, True))
    #basic_worker(handler, retrieve_blob=True)
    p.start()
    p.join()


basic_worker(handler, retrieve_blob=True)
