import modal
app = modal.App("probe-acts")
from modal_acts import image, data_vol, DATA_PATH, _setup_acts_env

@app.function(image=image, volumes={DATA_PATH: data_vol}, timeout=300)
def probe():
    _setup_acts_env()
    import acts, acts.examples
    names = [n for n in dir(acts.examples) if 'Handle' in n or 'Board' in n or 'Track' in n]
    print('Handle/Board/Track:', names)
    print('has ReadDataHandle', hasattr(acts.examples, 'ReadDataHandle'))
    print('has ConstTrackContainer', hasattr(acts.examples, 'ConstTrackContainer'))
    # WhiteBoard methods
    import inspect
    wb = acts.examples.WhiteBoard
    print('WhiteBoard attrs', [a for a in dir(wb) if not a.startswith('_')])

if __name__ == '__main__':
    with app.run():
        probe.remote()
