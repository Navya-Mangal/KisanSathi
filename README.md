# 🌾 Kisan Saathi
Hindi-first PWA-style app for Rajasthan farmers: crop disease detection (MobileNetV3), mandi price lookup (data.gov.in), grounded advisory.

## Run
```bash
pip install -r backend/requirements.txt
export DATA_GOV_API_KEY=your_key      # free from data.gov.in
uvicorn backend.main:app --reload     # open http://localhost:8000
```
## Train model
1. Download ORIGINAL PlantVillage (not augmented) -> split 70/15/15 into `data/{train,val,test}/<class>/`
2. `python ml/train.py --data data`  (creates backend/model.pt + classes.json)
3. Evaluate on PlantDoc / your own field photos; record lab-vs-field gap here.

## Roadmap
- [ ] Wheat + mustard datasets, local khet photos (verified by KVK)
- [ ] Mandi history table (daily cron) + forecast (baseline vs Prophet/XGBoost)
- [ ] RAG advisory over ICAR/SKNAU PDFs (FAISS) with sources
- [ ] Auth, scan history, SMS alerts, voice input, Docker deploy (Render)
- [ ] Results table: lab acc / field acc / farmer feedback
