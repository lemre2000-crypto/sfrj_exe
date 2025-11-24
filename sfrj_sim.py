import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# --- 1. FİZİK VE HESAPLAMA FONKSİYONLARI ---
def get_flight_conditions(altitude, mach, gamma=1.4):
    T0, P0, g, R, a = 288.15, 101325.0, 9.80665, 287.05, -0.0065
    if altitude <= 11000:
        T_static = T0 + a * altitude
        P_static = P0 * (T_static / T0)**(-g / (R * a))
    else:
        T11, P11 = 216.65, 22632.1
        T_static, P_static = T11, P11 * np.exp(-g * (altitude - 11000) / (R * T11))
    T_total = T_static * (1 + (gamma - 1) / 2 * mach**2)
    P_total = P_static * (1 + (gamma - 1) / 2 * mach**2)**(gamma / (gamma - 1))
    return T_total, P_total, T_static, P_static

def run_simulation(inputs):
    alt, mach, mdot_total, bpr = inputs['alt'], inputs['mach'], inputs['mdot_total'], inputs['bpr']
    A_coeff, n_flux, m_pres, t_temp = inputs['A_coeff'], inputs['n_flux'], inputs['m_pres'], inputs['t_temp']
    L_fuel, D_port, D_inlet, D_outer, D_throat = inputs['L_fuel']/1000, inputs['D_port']/1000, inputs['D_inlet']/1000, inputs['D_outer']/1000, inputs['D_throat']/1000
    rho_fuel, gamma_gas, R_gas = inputs['rho_fuel'], 1.25, 287.0
    
    Tt_air, Pt_air, _, _ = get_flight_conditions(alt, mach)
    A_throat = np.pi * (D_throat**2) / 4
    mdot_combustor = mdot_total / (1 + bpr)
    mdot_bypass = mdot_total - mdot_combustor
    
    # Lookup Table
    map_AFR_x = np.array([4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 100])
    map_AFR_y = np.array([2500, 2500, 2750, 2820, 2800, 2750, 2650, 2580, 2500, 2400, 2310, 2250, 2180, 500])
    flow_const = np.sqrt(gamma_gas/R_gas) * ((gamma_gas+1)/2)**(-(gamma_gas+1)/(2*(gamma_gas-1)))

    dt, time_max, time, Pt4, sim_running = 0.1, 200, 0, 500000, True
    history = {'time': [], 'D_port': [], 'Pt4': [], 'r_dot': [], 'AFR': [], 'mdot_fuel': [], 'T_mixed': []}
    msg = "Simülasyon Tamamlandı"

    while sim_running and time < time_max:
        A_port = np.pi * (D_port**2) / 4
        A_burn_surface = np.pi * D_port * L_fuel
        
        # Basınç İterasyonu
        p_iter = 0
        while p_iter < 50: # Basitleştirilmiş loop
            p_iter += 1
            G_ox = mdot_combustor / A_port
            r_dot = A_coeff * (G_ox**n_flux) * ((Pt4*1e-5)**m_pres) * (Tt_air**t_temp) * 0.01
            mdot_fuel = rho_fuel * A_burn_surface * r_dot
            AFR = mdot_combustor / mdot_fuel
            T_comb = np.interp(AFR, map_AFR_x, map_AFR_y)
            mdot_nozzle = mdot_combustor + mdot_fuel + mdot_bypass
            T_mixed = ((mdot_combustor+mdot_fuel)*T_comb + mdot_bypass*Tt_air)/mdot_nozzle
            Pt4_new = mdot_nozzle * np.sqrt(T_mixed) / (A_throat * flow_const)
            if abs(Pt4_new - Pt4)/Pt4 < 0.005: break
            Pt4 = Pt4_new

        history['time'].append(time)
        history['D_port'].append(D_port*1000)
        history['Pt4'].append(Pt4*1e-5)
        history['r_dot'].append(r_dot*1000)
        history['AFR'].append(AFR)
        history['mdot_fuel'].append(mdot_fuel)
        history['T_mixed'].append(T_mixed)
        
        D_port += 2 * r_dot * dt
        time += dt
        if D_port >= D_outer:
            sim_running = False
            msg = "Yakıt Tükendi"
            
    return history, Pt_air*1e-5, msg

# --- 2. GUI ---
class App:
    def __init__(self, root):
        self.root = root
        self.root.title("SFRJ Simulator")
        self.root.geometry("1000x700")
        
        # Sol Panel (Girdiler)
        frame_in = ttk.LabelFrame(root, text="Parametreler", padding=10)
        frame_in.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        
        self.ents = {}
        params = [
            ('İrtifa (m)', 'alt', 10000), ('Mach', 'mach', 2.5), 
            ('Hava Debisi (kg/s)', 'mdot_total', 3.0), ('Bypass Oranı', 'bpr', 1.0),
            ('Yakıt Boyu (mm)', 'L_fuel', 1300), ('Port Çapı (mm)', 'D_port', 147),
            ('Dış Çap (mm)', 'D_outer', 175), ('Giriş Çapı (mm)', 'D_inlet', 97),
            ('Boğaz Çapı (mm)', 'D_throat', 90), ('Yoğunluk', 'rho_fuel', 1500),
            ('Katsayı a', 'A_coeff', 0.0000445), ('Akı üssü n', 'n_flux', 0.53),
            ('Basınç üssü m', 'm_pres', 0.33), ('Sıcaklık üssü t', 't_temp', 0.71)
        ]
        
        for i, (lbl, key, val) in enumerate(params):
            ttk.Label(frame_in, text=lbl).grid(row=i, column=0, sticky='w')
            e = ttk.Entry(frame_in, width=10)
            e.insert(0, str(val))
            e.grid(row=i, column=1, pady=2)
            self.ents[key] = e
            
        ttk.Button(frame_in, text="HESAPLA", command=self.run).grid(row=15, columnspan=2, pady=10, sticky='ew')
        
        # Sağ Panel (Çıktılar)
        self.nb = ttk.Notebook(root)
        self.nb.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        self.txt = tk.Text(self.nb)
        self.nb.add(self.txt, text='Rapor')
        
        self.fig, self.axs = plt.subplots(2,2, figsize=(8,6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.nb)
        self.nb.add(self.canvas.get_tk_widget(), text='Grafik')

    def run(self):
        try:
            inp = {k: float(v.get()) for k, v in self.ents.items()}
            hist, Pt1, msg = run_simulation(inp)
            
            # Rapor
            self.txt.delete(1.0, tk.END)
            res = f"Durum: {msg}\nSüre: {hist['time'][-1]:.2f}s\n"
            res += f"Pt4: {hist['Pt4'][0]:.2f} -> {hist['Pt4'][-1]:.2f} Bar\n"
            res += f"Giriş Basıncı (Pt1): {Pt1:.2f} Bar\n"
            if hist['Pt4'][-1] >= Pt1: res += "⚠️ UYARI: BUZZ RİSKİ!\n"
            else: res += "✅ DURUM NORMAL\n"
            self.txt.insert(tk.END, res)
            
            # Grafik
            titles = ['Port (mm)', 'Basınç (Bar)', 'Hız (mm/s)', 'AFR']
            keys = ['D_port', 'Pt4', 'r_dot', 'AFR']
            for ax, key, title in zip(self.axs.flat, keys, titles):
                ax.clear()
                ax.plot(hist['time'], hist[key])
                ax.set_title(title)
                ax.grid(True)
            self.canvas.draw()
            self.nb.select(1) # Grafiğe geç
            
        except Exception as e:
            messagebox.showerror("Hata", str(e))

if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()
