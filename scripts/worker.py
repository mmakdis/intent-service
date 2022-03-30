import time
import io
import json
import logging.config
from multiprocessing import Process, Queue
from dotenv import load_dotenv
load_dotenv()

from jobqueue_worker import Job, Result, ResultStatus, basic_worker
from jobqueue_worker.config import loggers

logging.config.dictConfig(loggers.logger_config)

import logging

LOG = logging.getLogger(__name__.split(".")[0])

# one job success on job fails
job_success = False


def handler(job: Job, stream: io.BytesIO):
    global job_success
    job_success = not job_success

    if not job_success:
        LOG.info("Handled job as failed")
        return Result(status=ResultStatus.FAILED)

    result = Result(
        status=ResultStatus.SUCCESS,
        params={
            "worker": "Job Worker",
            "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "input_data": json.loads(job.parameters),
        },
        blob_name=job.id,
        blob_data=stream,
    )

    LOG.info("Handled job")
    print(result)
    return result


def run_worker():
    """Run basic_worker in a different process.
    """
    queue = Queue()
    p = Process(target=basic_worker, args=(handler, True))
    #basic_worker(handler, retrieve_blob=True)
    p.start()
    p.join()
