# Data directory

The included synthetic CSV is generated locally with:

```powershell
python src/generate_data.py
```

The repository includes the small synthetic CSV used by the demo. Large downloaded third-party datasets should not be committed; document their source and download instructions here instead.

Expected input columns:

| Column | Type | Description |
|---|---|---|
| `date` | date | Observation date |
| `sales` | numeric | Units sold or demand |
| `store` | optional categorical | Store identifier |
| `item` | optional categorical | Product identifier |
