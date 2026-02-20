import pandas as pd
import tkinter as tk
from tkinter import ttk, messagebox
import database

class DatabaseViewerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Disease Prediction Database Viewer (MySQL)")
        self.root.geometry("1000x500")

        self.button_frame = tk.Frame(self.root)
        self.button_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)

        self.btn_refresh = tk.Button(self.button_frame, text="Refresh Data", command=self.load_data, bg="#dddddd")
        self.btn_refresh.pack(side=tk.LEFT, padx=5)

        self.btn_clear = tk.Button(self.button_frame, text="Clear History", command=self.clear_history, bg="red", fg="white")
        self.btn_clear.pack(side=tk.LEFT, padx=5)

        self.tree_frame = tk.Frame(self.root)
        self.tree_frame.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)

        self.tree = ttk.Treeview(self.tree_frame, columns=('ID', 'Timestamp', 'Disease', 'Result', 'Probability'), show='headings')
        
        self.tree.heading('ID', text='ID')
        self.tree.heading('Timestamp', text='Timestamp')
        self.tree.heading('Disease', text='Disease Type')
        self.tree.heading('Result', text='Prediction')
        self.tree.heading('Probability', text='Probability (%)')

        self.tree.column('ID', width=50)
        self.tree.column('Timestamp', width=150)
        self.tree.column('Disease', width=150)
        self.tree.column('Result', width=150)
        self.tree.column('Probability', width=100)

        self.scrollbar = ttk.Scrollbar(self.tree_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscroll=self.scrollbar.set)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree.pack(expand=True, fill=tk.BOTH)

        self.load_data()

    def load_data(self):
        for i in self.tree.get_children():
            self.tree.delete(i)
        
        try:
            conn = database.get_db_connection(database.DB_NAME)
            
            df = pd.read_sql_query("SELECT * FROM predictions ORDER BY timestamp DESC", conn)
            conn.close()

            if not df.empty:
                for index, row in df.iterrows():
                    self.tree.insert("", tk.END, values=(
                        row['id'], 
                        row['timestamp'], 
                        row['disease_type'], 
                        row['prediction_result'], 
                        row['probability']
                    ))
            else:
                 pass
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load data: {e}")

    def clear_history(self):
        if messagebox.askyesno("Confirm Delete", "Are you sure you want to clear the entire history?"):
            try:
                database.clear_history()
                self.load_data()
                messagebox.showinfo("Success", "History cleared successfully.")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to clear history: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = DatabaseViewerApp(root)
    root.mainloop()
