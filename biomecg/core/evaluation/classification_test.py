# ================================================================================= #

import torch
import pandas as pd
from tqdm import tqdm
from biomecg.core.utils.tools import *

class ClassificationTestSingleBeat:
    """ To be instanciated with the test dataloader of interest (normal, cross-session,...)
    Specify a dataloader_name to differentiate results files.
    """
    def __init__(self, config: dict, dataloader, model, dataloader_name=""):
        self.config = config
        self.model = model
        self.dataloader = dataloader
        self.dataloader_name = dataloader_name
        self.device = torch.device(config["training"]["device"])
        self.model.network.to(self.device)
        self.model.network.eval()
        base_path = get_base_path()
        self.path_results = base_path + f"/results/{self.config['network']}/inference/{self.config['name']}/{self.dataloader_name}/results_{self.dataloader_name}.csv"
        create_dir(self.path_results)
        self.results = {
            "Label": [],
            "Probability": [],
            "pid1": [],
            "pid2": [],
            "s1": [],
            "b1": [],
            "s2": [],
            "b2": [],
            "t1": [],
            "t2": []
        }


    def evaluate(self):
        with torch.no_grad():
            for batch in tqdm(self.dataloader, desc="Evaluating"):
                # tests specified only in the xsession test dataloader
                b1, b2, y_batch, pid1s, s1s, b1s, pid2s, s2s, b2s, *tests = batch
                b1, b2 = b1.to(self.device), b2.to(self.device)
                y_batch = y_batch.to(self.device).float().unsqueeze(1)
                x_concat = torch.cat([b1, b2], dim=1)
                logits = self.model.network(x_concat)
                prob = torch.sigmoid(logits)
                self.results["Label"].extend(y_batch.cpu().numpy().flatten().tolist())
                self.results["Probability"].extend(prob.cpu().numpy().flatten().tolist())
                self.results["pid1"].extend(pid1s)
                self.results["s1"].extend([v.item() for v in s1s]) #s1s,... are list of tensors0D
                self.results["b1"].extend([v.item() for v in b1s])
                self.results["pid2"].extend(pid2s)
                self.results["s2"].extend([v.item() for v in s2s])
                self.results["b2"].extend([v.item() for v in b2s])
                
                if tests:
                    t1s, t2s = tests
                    self.results["t1"].extend([v.item() for v in t1s])
                    self.results["t2"].extend([v.item() for v in t2s])
                else:
                    batch_size = b1.size(0)
                    t1s = [None] * batch_size
                    t2s = [None] * batch_size
                    self.results["t1"].extend(t1s)
                    self.results["t2"].extend(t2s)
        self.save_results()


    def save_results(self):
        df = pd.DataFrame(self.results)
        df.to_csv(self.path_results, index=False)
        print(f"\n# --- Results saved to: {self.path_results} --- #")
