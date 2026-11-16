"""Render paper figures from aggregate results; no training or inference."""
from pathlib import Path
import nbformat
from nbclient import NotebookClient
from biomecg.paths import ROOT
def main():
 p=ROOT/'notebooks/paper_figs_and_charts.ipynb'
 n=nbformat.read(p,4)
 NotebookClient(n,timeout=300,resources={'metadata':{'path':str(ROOT)}}).execute()
 out=ROOT/'outputs/paper';out.mkdir(parents=True,exist_ok=True)
 nbformat.write(n,out/p.name)
if __name__=='__main__':main()
