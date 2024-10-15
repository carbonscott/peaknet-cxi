import torch
import logging
import time
from psana_ray.data_reader import DataReader, DataReaderError
from torch.utils.data import IterableDataset

class QueueDataset(IterableDataset):
    def __init__(self):
        super().__init__()
        self.reader = None

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  # single-process data loading
            return self.data_iterator()
        else:  # in a worker process
            return self.data_iterator(worker_id=worker_info.id, num_workers=worker_info.num_workers)

    def data_iterator(self, worker_id=0, num_workers=1):
        if self.reader is None:
            self.reader = DataReader()
            self.reader.connect()

        try:
            while True:
                try:
                    data = self.reader.read()
                    if data is None:
                        logging.debug("No data received, sleeping...")
                        time.sleep(0.1)  # Short sleep to avoid busy-waiting
                        continue
                    rank, idx, image_data = data
                    if idx % num_workers == worker_id:
                        logging.debug(f"Worker {worker_id}: Received data: rank={rank}, idx={idx}, image_shape={image_data.shape}")
                        tensor = torch.from_numpy(image_data).unsqueeze(0)  # (H,W) -> (1,H,W)
                        yield tensor
                except DataReaderError as e:
                    logging.error(f"DataReader error: {e}")
                    break
                except Exception as e:
                    logging.error(f"Unexpected error in QueueDataset: {e}")
                    time.sleep(1)  # Longer sleep on unexpected errors
        finally:
            if self.reader:
                self.reader.close()
                logging.debug(f"Worker {worker_id}: DataReader closed")

    def __del__(self):
        if self.reader:
            self.reader.close()
            logging.debug("DataReader closed in destructor")
