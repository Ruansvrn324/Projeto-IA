import numpy as np
import plotly.express as px

dados = np.loadtxt(r"F:\Arquivos de Programas\Python projects\arsenio_dataset (1).csv", delimiter=",", skiprows=1)

x1 = dados[:,0]#Idade
x2 = dados[:,2]#Uso_Beber
x3 = dados[:,3]#Uso_Cozinhar
x4 = dados[:,4]#Arsenio_Agua
y  = dados[:,5]#Arsenio_Unhas
n = len(x1)
x = np.column_stack((np.ones(n), x1, x2, x3, x4))


x_T = x.T
B = np.dot(x_T, y)
beta = np.linalg.inv(x_T @ x) @ B
print("Coeficientes:")
print(beta)
p = len(beta)-1

y_pred = x @ beta
ss_total = np.sum((y-y.mean())**2)
ss_res = np.sum((y-y_pred)**2)
r_squared = 1- ss_res/ss_total
print("Coeficiente de determinação R²:")
print(r_squared)

r2_adj = 1-(((1-r_squared)*(n-1))/(n-p-1))
print("Coeficiente de determinação ajustado R²aj:")
print(r2_adj)

rmse = np.sqrt(np.sum((y-y_pred)**2)/n)
print("RMSE:")
print(rmse)

mae = np.sum(np.abs(y-y_pred))/n
print("MAE:")
print(mae)

equacao = f"y = {beta[0]:.3f} + {beta[1]:.3f}*x1 + {beta[2]:.3f}*x2 + {beta[3]:.3f}*x3 + {beta[4]:.3f}*x4"
print(equacao)

i = 0
while i <= n-1:
    tabela_residuos = f"Observação i: {i+1};\tyi: {y[i]};\ty da previsão: {y_pred[i]};\tResíduo: {y[i]-y_pred[i]}"
    print(tabela_residuos)
    i += 1

beta[0] = 0
y_pred = x @ beta
ss_res = np.sum((y-y_pred)**2)
r_squared = 1- ss_res/ss_total
print("Coeficiente de determinação R² com intercepto 0:")
print(r_squared)

rmse = np.sqrt(np.sum((y-y_pred)**2)/n)
print("RMSE com intercepto 0:")
print(rmse)

fig1 = px.scatter(x=x1, y=y, trendline="ols", labels={'x': 'Idade', 'y': 'Arsênio nas Unhas'},title="Relação entre Idade e Arsênio nas Unhas")
fig2 = px.scatter(x=x2, y=y, trendline="ols", labels={'x': 'Uso_Beber', 'y': 'Arsênio nas Unhas'},title="Relação entre Uso para Beber e Arsênio nas Unhas")
fig3 = px.scatter(x=x3, y=y, trendline="ols", labels={'x': 'Uso_Cozinhar', 'y': 'Arsênio nas Unhas'},title="Relação entre Uso para Cozinhar e Arsênio nas Unhas")
fig4 = px.scatter(x=x4, y=y, trendline="ols", labels={'x': 'Arsenio_Agua', 'y': 'Arsênio nas Unhas'},title="Relação entre Arsenio na Água e Arsênio nas Unhas")


fig1.show()
fig2.show()
fig3.show()
fig4.show()
