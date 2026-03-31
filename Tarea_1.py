# Tarea 1 - Simulación
# Eduardo Jaramillo, Eduardo Mariqueo, Vicente Ramirez

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# 1. Definición de Parámetros
rho = 0.5           # Tasa proliferación tumoral
mu = 0.2            # Tasa proliferación células sanas
K_T = 1e6           # Capacidad de carga tumoral
K_H = 1e6           # Capacidad de carga células sanas
beta = 1.5e-6       # Tasa de invasión tumoral sobre tejido sano
gamma = 0.5e-6      # Resistencia del tejido sano
alpha = 0.8         # Eficacia del tratamiento
delta = 0.1         # Toxicidad del fármaco sobre células sanas
lam = 0.3           # Tasa de eliminación del fármaco por el cuerpo

# 2. Condiciones Iniciales
H0 = 1e6            # Tejido sano al máximo
T0 = 1e4            # Tumor incipiente
C0 = 0.0            # Sin fármaco al inicio
y0 = [T0, H0, C0]

# 3. Vector de tiempo (100 días discretizados en 1000 puntos)
t = np.linspace(0, 100, 1000)

# 4. Sistema de Ecuaciones Diferenciales
def modelo_cancer(y, t, tipo_tratamiento):
    T, H, C = y

    # Lógica de dosificación según el escenario experimental
    if tipo_tratamiento == 1:
        dosis = 0.0                          # Control: Sin tratamiento
    elif tipo_tratamiento == 2:
        dosis = 0.5 if t >= 20 else 0.0      # Conservador: Dosis baja y constante desde el día 20
    elif tipo_tratamiento == 3:
        dosis = 2.0 if (t % 20) < 5 else 0.0 # Agresivo: Dosis alta por 5 días, cada 20 días (ciclos)

    # Ecuaciones
    dTdt = rho * T * (1 - T/K_T) - gamma * T * H - alpha * T * C
    dHdt = mu * H * (1 - H/K_H) - beta * H * T - delta * H * C
    dCdt = -lam * C + dosis

    return [dTdt, dHdt, dCdt]

# 5. Función para integrar y graficar
def simular_y_graficar(tipo, nombre_archivo, titulo):
    # Resolver las EDOs
    solucion = odeint(modelo_cancer, y0, t, args=(tipo,))
    T_sol = solucion[:, 0]
    H_sol = solucion[:, 1]
    C_sol = solucion[:, 2]

    # Configuración de la gráfica con doble eje Y
    fig, ax1 = plt.subplots(figsize=(9, 5))

    ax1.set_xlabel('Tiempo (días)')
    ax1.set_ylabel('Población Celular', color='black')
    line1, = ax1.plot(t, T_sol, label='Células Tumorales (T)', color='red', linewidth=2)
    line2, = ax1.plot(t, H_sol, label='Células Sanas (H)', color='blue', linewidth=2)
    ax1.tick_params(axis='y', labelcolor='black')

    # Segundo eje para la concentración del fármaco (valores mucho menores)
    ax2 = ax1.twinx()  
    ax2.set_ylabel('Concentración del Fármaco (C)', color='green')  
    line3, = ax2.plot(t, C_sol, label='Dosis (C)', color='green', linestyle='--')
    ax2.tick_params(axis='y', labelcolor='green')

    plt.title(titulo)
    fig.legend(handles=[line1, line2, line3], loc='center right', bbox_to_anchor=(0.85, 0.5))
    ax1.grid(True, linestyle=':', alpha=0.7)
    fig.tight_layout()
    
    # Guardar la imagen
    plt.savefig(nombre_archivo, dpi=300)
    plt.close()

# 6. Ejecución de los Escenarios
print("Iniciando simulación...")
simular_y_graficar(1, 'escenario1.png', 'Escenario 1: Progresión natural (Sin tratamiento)')
simular_y_graficar(2, 'escenario2.png', 'Escenario 2: Tratamiento conservador constante')
simular_y_graficar(3, 'escenario3.png', 'Escenario 3: Tratamiento agresivo cíclico')
print("¡Simulación completada! Las gráficas se han guardado con éxito.")