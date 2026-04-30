# Sistema de Análisis Electoral ONPE 2021

## Segunda Elección Presidencial — 6 de junio de 2021

### Dataset oficial
- Fuente: ONPE — datosabiertos.gob.pe
- 86,488 mesas de sufragio
- Pedro Castillo (Perú Libre) vs Keiko Fujimori (Fuerza Popular)
- Resultado: Castillo 50.126% — Fujimori 49.874%

### Tecnologías
- Python 3.11 + Streamlit + Pandas + Plotly + Scikit-learn
- Docker para despliegue
- GitHub para control de versiones

### Ejecutar localmente
pip install -r requirements.txt
streamlit run app.py

### Ejecutar con Docker
docker build -t onpe-electoral-2021 .
docker run -p 8501:8501 onpe-electoral-2021
