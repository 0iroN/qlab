"""Orbital penceresi testleri.

Bu dosyadaki testler gerçek bir ekran olmadan koşar (QT_QPA_PLATFORM=offscreen).
Offscreen kipte OpenGL bağlamı oluşturulamayabilir; o yüzden çizimin kendisi
değil, veri hattı ve pencere durumu doğrulanır. Gerçek görsel doğrulama
insanın işidir ve README'de böyle yazılıdır.
"""

from __future__ import annotations

import numpy as np
import pytest


def test_pyqtgraph_opengl_import_edilebiliyor():
    """PyOpenGL kurulu değilse bu modülün tamamı çalışmaz.

    Bilinçli olarak `importorskip` DEĞİL: PyOpenGL `pyproject.toml`'da
    bağımlılıktır, yani yokluğu bozuk bir ortam demektir. Bu depo
    `tests/hf/conftest.py`'de aynı gerekçeyle `pytest.skip` yerine
    `pytest.fail` kullanır — sessizce atlanan bir doğrulama, doğrulama
    değildir.
    """
    import pyqtgraph.opengl as gl

    assert hasattr(gl, "GLViewWidget")
    assert hasattr(gl, "GLMeshItem")


def test_isosurface_kure_uzerinde_calisiyor():
    """pg.isosurface marching cubes'tur ve saf NumPy'dir; OpenGL gerektirmez."""
    import pyqtgraph as pg

    x, y, z = np.mgrid[-1:1:20j, -1:1:20j, -1:1:20j]
    data = np.ascontiguousarray(x * x + y * y + z * z, dtype=np.float32)
    vertices, faces = pg.isosurface(data, 0.5)
    assert len(vertices) > 100
    assert len(faces) > 100
    assert faces.shape[1] == 3
