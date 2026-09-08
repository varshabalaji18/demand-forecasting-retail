from streamlit.testing.v1 import AppTest
from pathlib import Path

def test_app_horizons_and_filters():
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / 'app.py', default_timeout=60).run()
    assert not app.exception and not app.error
    for horizon in [30,60,90]:
        app.select_slider[0].set_value(horizon).run()
        assert not app.exception and not app.error
        assert app.metric[0].value == f'{horizon} days'
        assert len(app.get('plotly_chart')) == 1
        assert len(app.get('download_button')) == 2
    app.selectbox[0].select(1).run()
    app.selectbox[1].select(1).run()
    assert not app.exception and not app.error
