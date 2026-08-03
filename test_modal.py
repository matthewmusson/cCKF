"""Test Modal with multi-stage Dockerfile (clean Ubuntu base + spack from ghcr.io)."""

import modal

app = modal.App("surp-test")

image = (
    modal.Image.from_dockerfile("Dockerfile.modal", add_python="3.13")
    .pip_install("pyyaml")
)


@app.function(image=image, timeout=120)
def test_python():
    import sys
    print(f"Python: {sys.executable}")
    print(f"Version: {sys.version}")

    # Test acts import
    sys.path.insert(0, "/opt/acts_pypath")
    with open("/etc/spack_site_packages") as f:
        for line in f:
            p = line.strip()
            if p:
                sys.path.append(p)

    # Source env
    import os
    import subprocess
    result = subprocess.run(
        ["bash", "-c", "source /etc/acts_env.sh && env"],
        capture_output=True, text=True,
    )
    for line in result.stdout.splitlines():
        if "=" in line:
            key, _, val = line.partition("=")
            os.environ[key] = val

    os.environ["ODD_PATH"] = "/opt/ODD_v5/install/share/OpenDataDetector"

    try:
        import acts
        print(f"ACTS imported: {acts.__file__}")
        from acts.examples.odd import getOpenDataDetector
        print("ODD geometry: accessible")
    except Exception as e:
        print(f"ACTS import error: {e}")

    return "OK"


@app.local_entrypoint()
def main():
    result = test_python.remote()
    print(f"Result: {result}")
