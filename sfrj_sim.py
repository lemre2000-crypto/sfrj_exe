import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# ---------------------------------------------------------
# 1. FİZİK VE HESAPLAMA FONKSİYONLARI
# ---------------------------------------------------------

def get_flight_conditions(altitude, mach, gamma=1.4):
    T0 = 288.15
    P0 = 101325.0
    g = 9.80665
    R = 287.05
    a = -0.0065 

    if altitude <= 11000:
        T_static = T0 + a * altitude
        P_static = P0 * (T_static / T0)**(-g / (R * a))
    else:
        T11 = 216.65
        P11 = 22632.1
        T_static = T11
        P_static = P11 * np.exp(-g * (altitude - 11000) / (R * T11))

    T_total = T_static * (1 + (gamma - 1) / 2 * mach**2)
    P_total = P_static * (1 + (gamma - 1) / 2 * mach**2)**(gamma / (gamma - 1))

    return T_total, P_total, T_static, P_static

def run_simulation(inputs):
    # Girdileri paketinden çıkar
    alt = inputs['alt']
    mach = inputs['mach']
    mdot_total = inputs['mdot_total']
    bpr = inputs['bpr']
    A_coeff = inputs['A_coeff']
    n_flux = inputs['n_flux']
    m_pres = inputs['m_pres']
    t_temp = inputs['t_temp']
    
    L_fuel = inputs['L_fuel'] / 1000.0
    D_port = inputs['D_port'] / 1000.0
    D_inlet = inputs['D_inlet'] / 1000.0
    D_outer = inputs['D_outer'] / 1000.0
    D_throat = inputs['D_throat'] / 1000.0
    rho_fuel = inputs['rho_fuel']

    gamma_air = 1.4
    gamma_gas = 1.25 
    R_gas = 287.0    

    Tt_air, Pt_air, _, _ = get_flight_conditions(alt, mach, gamma_air)
    Pt_air_bar = Pt_air * 1e-5

    A_throat = np.pi * (D_throat**2) / 4
    
    mdot_combustor = mdot_total / (1 + bpr)
    mdot_bypass = mdot_total - mdot_combustor

    map_AFR_x = np.array([4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 100])
    map_AFR_y = np.array([2500, 2500, 2750, 2820, 2800, 2750, 2650, 2580, 2500, 2400, 2310, 2250, 2180, 500])

    flow_const = np.sqrt(gamma_gas/R_gas) * ((gamma_gas+1)/2)**(-(gamma_gas+1)/(2*(gamma_gas-1)))

    dt = 0.1
    time_max = 200
    time = 0
    sim_running = True
    Pt4 = 500000 # 5 Bar başlangıç

    history = {'time': [], 'D_port': [], 'Pt4': [], 'r_dot': [], 'AFR': [], 'mdot_fuel': [], 'T_mixed': []}
    status_msg = "Simülasyon başarıyla tamamlandı."

    while sim_running and time < time_max:
        A_port = np.pi * (D_port**2) / 4
        A_burn_surface = np.pi * D_port * L_fuel

        p_error = 1.0
        p_iter = 0
        while p_error > 0.001 and p_iter < 100:
            p_iter += 1
            G_ox = mdot_combustor / A_port
            r_dot = A_coeff * (G_ox**n_flux) * ((Pt4*1e-5)**m_pres) * (Tt_air**t_temp) 
            r_dot = r_dot * 0.01 
            mdot_fuel = rho_fuel * A_burn_surface * r_dot
            AFR = mdot_combustor / mdot_fuel
            T_comb = np.interp(AFR, map_AFR_x, map_AFR_y)
            mdot_nozzle = mdot_combustor + mdot_fuel + mdot_bypass
            mdot_core = mdot_combustor + mdot_fuel
            T_mixed = ((mdot_core * T_comb) + (mdot_bypass * Tt_air)) / mdot_nozzle
            Pt4_new = mdot_nozzle * np.sqrt(T_mixed) / (A_throat * flow_const)
            p_error = abs(Pt4_new - Pt4) / Pt4
            Pt4 = Pt4_new

        history['time'].append(time)
        history['D_port'].append(D_port * 1000)
        history['Pt4'].append(Pt4 * 1e-5)
        history['r_dot'].append(r_dot * 1000)
        history['AFR'].append(AFR)
        history['mdot_fuel'].append(mdot_fuel)
        history['T_mixed'].append(T_mixed)

        dr = r_dot * dt
        D_port += 2 * dr
        time += dt

        if D_port >= D_outer:
            sim_running = False
            status_msg = "Yakıt Tükendi (Dış Çapa Ulaşıldı)."

    return history, Pt_air_bar, status_msg

# ---------------------------------------------------------
# 2. GELİŞMİŞ ARAYÜZ (GUI)
# ---------------------------------------------------------

class SfrjApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🚀 SFRJ Performans Simülatörü v2.0")
        self.root.geometry("900x550")
        self.root.configure(bg="#f0f0f0") # Hafif gri arka plan

        # --- Stil Ayarları ---
        style = ttk.Style()
        style.theme_use('clam')
        style.configure("TLabelframe", background="#f0f0f0", font=('Arial', 10, 'bold'))
        style.configure("TLabelframe.Label", background="#f0f0f0", foreground="#333")
        style.configure("TLabel", background="#f0f0f0", font=('Arial', 9))
        style.configure("TEntry", fieldbackground="white", font=('Arial', 10))

        # Ana Başlık
        header = tk.Label(root, text="SFRJ Tasarım ve Analiz Aracı", bg="#f0f0f0", font=("Segoe UI", 16, "bold"), fg="#2c3e50")
        header.pack(pady=15)

        # Girdileri tutacak ana frame
        main_frame = tk.Frame(root, bg="#f0f0f0")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20)

        # Sözlük oluştur (girdileri saklamak için)
        self.entries = {}

        # --- 1. Sütun: Uçuş Koşulları ---
        frame_flight = ttk.LabelFrame(main_frame, text=" ✈️ Uçuş Koşulları ", padding=15)
        frame_flight.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        
        self.create_row(frame_flight, "İrtifa (m):", "alt", 10000, 0)
        self.create_row(frame_flight, "Mach Sayısı:", "mach", 2.5, 1)
        self.create_row(frame_flight, "Toplam Hava (kg/s):", "mdot_total", 3.0, 2)
        self.create_row(frame_flight, "Bypass Oranı (BPR):", "bpr", 1.0, 3)

        # --- 2. Sütun: Geometri ---
        frame_geo = ttk.LabelFrame(main_frame, text=" 📐 Geometri ", padding=15)
        frame_geo.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)

        self.create_row(frame_geo, "Yakıt Uzunluğu (mm):", "L_fuel", 1300, 0)
        self.create_row(frame_geo, "Port Çapı (mm):", "D_port", 147, 1)
        self.create_row(frame_geo, "Yakıt Dış Çapı (mm):", "D_outer", 175, 2)
        self.create_row(frame_geo, "YO Giriş Çapı (mm):", "D_inlet", 97, 3)
        self.create_row(frame_geo, "Lüle Boğazı (mm):", "D_throat", 90, 4)

        # --- 3. Sütun: Yakıt ve Katsayılar ---
        frame_fuel = ttk.LabelFrame(main_frame, text=" 🔥 Yakıt & Model ", padding=15)
        frame_fuel.grid(row=0, column=2, sticky="nsew", padx=10, pady=10)

        self.create_row(frame_fuel, "Yoğunluk (kg/m³):", "rho_fuel", 1500, 0)
        self.create_row(frame_fuel, "Katsayı (a):", "A_coeff", 0.0000445, 1)
        self.create_row(frame_fuel, "Akı Üssü (n):", "n_flux", 0.53, 2)
        self.create_row(frame_fuel, "Basınç Üssü (m):", "m_pres", 0.33, 3)
        self.create_row(frame_fuel, "Sıcaklık Üssü (t):", "t_temp", 0.71, 4)

        # Sütun ağırlıklarını eşitle (Genişlesinler)
        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.columnconfigure(2, weight=1)

        # --- Başlat Butonu ---
        btn_start = tk.Button(root, text="SİMÜLASYONU BAŞLAT", font=("Segoe UI", 12, "bold"), 
                              bg="#27ae60", fg="white", activebackground="#2ecc71", activeforeground="white",
                              cursor="hand2", command=self.start_simulation)
        btn_start.pack(fill=tk.X, padx=30, pady=25, ipady=5)

        # Alt bilgi
        footer = tk.Label(root, text="Oluşturulan pencereler simülasyon sonuçlarını içerecektir.", bg="#f0f0f0", fg="#7f8c8d")
        footer.pack(pady=5)

    def create_row(self, parent, label_text, key, default_val, row_idx):
        lbl = ttk.Label(parent, text=label_text)
        lbl.grid(row=row_idx, column=0, sticky="w", pady=5)
        
        ent = ttk.Entry(parent, width=12, justify="center")
        ent.insert(0, str(default_val))
        ent.grid(row=row_idx, column=1, sticky="e", pady=5, padx=5)
        
        self.entries[key] = ent

    def start_simulation(self):
        try:
            # 1. Verileri topla
            inputs = {key: float(entry.get()) for key, entry in self.entries.items()}
            
            # 2. Simülasyonu çalıştır
            hist, Pt_air_bar, msg = run_simulation(inputs)
            
            # 3. Sonuçları işle
            self.show_results_window(inputs, hist, Pt_air_bar, msg)
            self.show_plots_window(hist, Pt_air_bar)

        except ValueError:
            messagebox.showerror("Hata", "Lütfen tüm alanlara geçerli sayısal değerler giriniz.")
        except Exception as e:
            messagebox.showerror("Kritik Hata", f"Beklenmeyen bir hata oluştu:\n{str(e)}")

    def show_results_window(self, inputs, hist, Pt_air_bar, msg):
        win = tk.Toplevel(self.root)
        win.title("📝 Simülasyon Sonuç Raporu")
        win.geometry("500x700")
        
        txt = tk.Text(win, font=("Consolas", 10), padx=10, pady=10)
        txt.pack(fill=tk.BOTH, expand=True)

        total_fuel = sum(hist['mdot_fuel']) * 0.1
        T_mixed_final = hist['T_mixed'][-1]
        mdot_nozzle_final = inputs['mdot_total'] + hist['mdot_fuel'][-1]
        fuel_thickness = (inputs['D_outer'] - inputs['D_port']) / 2
        is_buzz = hist['Pt4'][-1] >= Pt_air_bar

        A_port_end = np.pi * (hist['D_port'][-1]/1000)**2 / 4 
        A_inlet = np.pi * (inputs['D_inlet']/1000)**2 / 4
        A_throat = np.pi * (inputs['D_throat']/1000)**2 / 4
        
        A_port_mm2 = A_port_end * 1e6
        A_inlet_mm2 = A_inlet * 1e6
        A_throat_mm2 = A_throat * 1e6

        report_str = f"======================================\n SONUÇ: {msg}\n======================================\n"
        report_str += f"Simülasyon Süresi   : {hist['time'][-1]:.2f} s\n"
        report_str += f"Toplam Yakıt        : {total_fuel:.4f} kg\n"
        report_str += f"Yakıt Et Kalınlığı  : {fuel_thickness:.4f} mm\n"
        report_str += "--------------------------------------\nANLIK PERFORMANS (SİMÜLASYON SONU):\n"
        report_str += f"Toplam Debi         : {mdot_nozzle_final:.4f} kg/s\n"
        report_str += f"Lüle Giriş Sıcaklığı: {T_mixed_final:.2f} K\n"
        report_str += f"AFR                 : {hist['AFR'][-1]:.2f}\n"
        report_str += f"Regresyon Hızı      : {hist['r_dot'][-1]:.4f} mm/s\n"
        report_str += "--------------------------------------\nBASINÇ DURUMU:\n"
        report_str += f"Giriş Basıncı (Pt1) : {Pt_air_bar:.2f} Bar\n"
        report_str += f"YO Basıncı (Pt4)    : {hist['Pt4'][0]:.2f} -> {hist['Pt4'][-1]:.2f} Bar\n\nDURUM ANALİZİ:\n"
        
        if is_buzz:
            report_str += "⚠️ UYARI: BUZZ RİSKİ (Pt4 > Pt1)\n   Motor kararsız çalışabilir (Unstart)."
        else:
            report_str += "✅ NORMAL ÇALIŞMA (Pt4 < Pt1)"

        report_str += f"\n--------------------------------------\nGEOMETRİK VERİLER (SON DURUM):\n"
        report_str += f"Port Çapı           : {hist['D_port'][0]:.1f} -> {hist['D_port'][-1]:.1f} mm\n\n"
        report_str += f"Port Çıkış Alanı    : {A_port_mm2:.1f} mm²\n"
        report_str += f"Giriş Alanı         : {A_inlet_mm2:.1f} mm²\n"
        report_str += f"Boğaz Alanı         : {A_throat_mm2:.1f} mm²\n\n"
        report_str += f"A_port / A_inlet    : {A_port_end / A_inlet:.4f}\n"
        report_str += f"A_port / A_throat   : {A_port_end / A_throat:.4f}\n"
        report_str += f"A_throat / A_inlet  : {A_throat / A_inlet:.4f}\n"
        report_str += "======================================\n"
        
        txt.insert(tk.END, report_str)
        txt.config(state=tk.DISABLED)

    def show_plots_window(self, hist, Pt_air_bar):
        win = tk.Toplevel(self.root)
        win.title("📊 Performans Grafikleri")
        win.geometry("900x700")

        fig, axs = plt.subplots(2, 2, figsize=(10, 8))
        
        axs[0, 0].plot(hist['time'], hist['D_port'], 'b-', lw=2)
        axs[0, 0].set_title('Port Çapı Değişimi')
        axs[0, 0].set_ylabel('Çap (mm)')
        axs[0, 0].grid(True)

        axs[0, 1].plot(hist['time'], hist['Pt4'], 'r-', lw=2)
        axs[0, 1].axhline(y=Pt_air_bar, color='k', linestyle='--', label='Giriş (Pt1)')
        axs[0, 1].set_title('Yanma Odası Basıncı')
        axs[0, 1].set_ylabel('Basınç (Bar)')
        axs[0, 1].legend()
        axs[0, 1].grid(True)

        axs[1, 0].plot(hist['time'], hist['r_dot'], 'g-', lw=2)
        axs[1, 0].set_title('Yakıt Yanma Hızı')
        axs[1, 0].set_xlabel('Zaman (s)')
        axs[1, 0].set_ylabel('r_dot (mm/s)')
        axs[1, 0].grid(True)

        axs[1, 1].plot(hist['time'], hist['AFR'], 'purple', lw=2)
        axs[1, 1].set_title('Hava / Yakıt Oranı (AFR)')
        axs[1, 1].set_xlabel('Zaman (s)')
        axs[1, 1].grid(True)

        fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=win)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)


if __name__ == "__main__":
    root = tk.Tk()
    app = SfrjApp(root)
    root.mainloop()
