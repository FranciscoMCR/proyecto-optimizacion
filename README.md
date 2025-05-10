# 🧠 Optimization Playground

Un entorno interactivo para experimentar con algoritmos de optimización clásicos y estocásticos. Implementado en Python con interfaz gráfica mediante Tkinter.

---

## 🚀 Características

- **Métodos clásicos:**
  - ✅ Gradient Descent (con o sin búsqueda lineal)
  - ✅ BFGS (cuasi-Newton)
  - ✅ Adam (optimizador adaptativo)

- **Métodos estocásticos:**
  - ✅ Stochastic Gradient Descent (SGD)

- **Búsqueda lineal:**
  - ✅ Armijo Backtracking
  - ✅ Wolfe Conditions

- **Visualización integrada:**
  - 📈 Convergencia de `f(x)` por iteración
  - 📈 Convergencia de `‖∇f(x)‖`
  - 📊 Tabla con métricas por iteración
  - 🌐 Visualización 3D para funciones con 2 variables

- **Métricas adicionales:**
  - ⏱️ Tiempo de ejecución
  - 🔢 Número de evaluaciones de `f(x)` y `∇f(x)`

---

## 🖥️ Interfaz Gráfica

La GUI permite:

- Ingresar funciones simbólicas como `x**2 + y**2`
- Especificar variables y punto inicial
- Configurar tolerancia, método, búsqueda lineal y tasa de aprendizaje
- Visualizar resultados gráficos y métricas detalladas
- Mostrar la función objetivo en 3D con `Show 3D Plot`

---

## 📦 Instalación

1. Clona el repositorio:

```bash
git clone https://github.com/tu-usuario/proyecto_optimizacion.git
cd proyecto_optimizacion
```
2. Crea y activa un entorno virtual:

```bash
python -m venv .venv
# Activar entorno:
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
```

3. Instala dependencias:

```bash
pip install -r requirements.txt
```

---

## ▶️ Ejecutar la aplicación
```bash
python main.py
```
