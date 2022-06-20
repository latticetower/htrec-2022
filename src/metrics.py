import numpy as np
import pywer

def compute_metrics(human_translations, system_translations, candidate_translations):
  def compute_cer_values(avalues, bvalues):
    # train_df["R1_CER"] = train_df.apply(lambda row: pywer.cer([row.HUMAN_TRANSCRIPTION], [R1]), axis=1)
    return np.asarray([pywer.cer([a], [b]) for a, b in zip(avalues, bvalues)])
    
  cer_values = compute_cer_values(human_translations, candidate_translations)
  print(f"Candidate CER: {cer_values.mean()}")
  # Computing the character error *reduction* rate (CERR)
  cer_ht_st = compute_cer_values(human_translations, system_translations)
  print(f"Candidate CERR: {(cer_ht_st - cer_values).mean()}")
  return cer_ht_st, cer_values