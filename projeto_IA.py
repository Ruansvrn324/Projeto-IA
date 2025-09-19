import numpy as np
import plotly.express as px

dados = np.loadtxt(r"F:\Arquivos de Programas\Python projects\arsenio_dataset (1).csv",skiprows=1,delimiter=",")

x1 = dados[:,0]#idade
x2 = dados[:,2]#uso_beber
x3 = dados[:,3]#uso_cizinhar
x4 = dados[:,4]#Arsenio_Agua
y = dados[:,5]#Arsenio_Unhas
n = len(x1)
X = np.column_stack((np.ones(n), x1, x2, x3, x4))


X_T = X.T
B = np.dot(X_T, y)
Beta = np.linalg.inv(X_T @ X) @ B 
p = len(Beta)-1

print("A)")
print("--- Coeficientes do modelo ajustado ---")
print(f"Intercepto: {Beta[0]:.3f}")
print(f"Coeficiente idade: {Beta[1]:.3f}")
print(f"Coeficiente uso ao beber: {Beta[2]:.3f}")
print(f"Coeficiente uso ao cozinhar: {Beta[3]:.3f}")
print(f"Coeficiente ursenio na agua: {Beta[4]:.3f}\n")

X_novo = np.array([1, 30, 5, 5, 0.135])
previsao = X_novo @ Beta

print("B)")
print("--- Previsão de arsenio nas unhas ---")
print("Valores de entrada:")
print(f"Idade: {X_novo[1]}")
print(f"Categoria para beber: {X_novo[2]}")
print(f"Categoria para cozinhar: {X_novo[3]}")
print(f"Arsênio na agua: {X_novo[4]}")
print(f"Concentração prevista de arsenio nas unhas: {previsao:.3f}\n")

ypred = X @ Beta
SStotal = np.sum((y-y.mean())**2)
SSresiduo = np.sum((y-ypred)**2)
RSquared = 1- SSresiduo/SStotal
print(f"Resíduo: {SSresiduo:.3f}")
print(f"Soma Total: {SStotal:.3f}")

if SStotal == 0:
    r2score = 1.0 if SSresiduo == 0 else 0.0
else:
    r2score = 1 - (SSresiduo / SStotal)
print("C)")
print("--- Performance do modelo ---")
print(f"O valor de R2 (RSquared) do modelo é: {r2score:.3f}")

print("D)")
r2_adj = 1-(((1-RSquared)*(n-1))/(n-p-1))
print(f"Coeficiente de determinação ajustado R²aj: {r2_adj:.3f}")
print("De fato, ao avaliarmos modelos de regressão múltipla, o R² Ajustado se destaca como uma ferramenta de avaliação mais robusta e confiável em comparação com o R² tradicional." \
" Este último apresenta uma peculiaridade que pode nos iludir: seu valor invariavelmente cresce cada vez que incorporamos uma nova variável ao modelo." \
" O R² Ajustado foi idealizado justamente para mitigar essa questão. Ele impõe uma espécie de 'penalidade' ao modelo ao serem inseridas variáveis que não contribuem de forma relevante para seu aprimoramento.")

x4 = dados[:,4]#Arsenio_Agua
y = dados[:,5]#Arsenio_Unhas

X4 = np.column_stack((np.ones(n), x4))
X_T = X4.T
Betax4 = np.linalg.inv(X_T @ X4) @ (X_T @y)
print("E)")
print("Resultados do Modelo Alternativo (Simples)")
print("--- Coeficientes do Modelo ---")
print(f"Intercepto: {Betax4[0]:.3f}")
print(f"Coeficiente 'Arsênio na Água': {Betax4[1]:.3f}")


Ypredx4 = X4 @ Betax4
SSresx4 = np.sum((y - Ypredx4)**2)
SStotx4 = np.sum((y - np.mean(y))**2)

r2 = 1 - (SSresx4 / SStotx4)
r2_ajustado = 1 - ((1 - r2) * (n - 1) / (n - p - 1))

print("--- Performance do Modelo ---")
print(f"R² Comum: {r2:.3f}")
print(f"R² Ajustado: {r2_ajustado:.3f}")


print("F)")
print("c)")
print("---Observação i---- Valor observado yi---- Valor ajustado ^yi---- Resíduo ei---")

i = 0
while i <= n-1:
    tabela_residuos = f"Observação i: {i+1};\tyi: {y[i]:.3f};\ty da previsão: {ypred[i]:.3f};\tResíduo: {y[i]-ypred[i]:.3f}"
    print(tabela_residuos)
    i += 1

print("G)")
Beta[0] = 0
y_pred = X @ Beta
SSres = np.sum((y-y_pred)**2)
r_squared = 1- SSres/SStotal
print(f"Coeficiente de determinação R² com intercepto 0:{r_squared:.3f}")
rmse = np.sqrt(np.sum((y-ypred)**2)/n)
print(f"RMSE: {rmse:.3f}")
rmse = np.sqrt(np.sum((y-y_pred)**2)/n)
print(f"RMSE com intercepto 0:{rmse:.3f}")
print("Impor uma intersecção zero seria válido apenas com uma justificativa teórica ou física irrefutável, que não existe aqui." \
    " A premissa de que a ausência de arsênio na água resulta em zero arsênio nas unhas é irrealista, dadas outras vias de exposição e níveis biológicos ou erros de medição." \
    " Portanto, fixar a intersecção em zero não tem uso no mundo real neste caso." \
    "mas, tbm leva a dados incerto e pouco precisos, demostrado ao ver que o RMSE sem o intercepto é mais que o dobro do RMSE com intercepto, a fins de ter um dado mais preciso da tabela usada, eu escolheria o RMSE com o intercepto")

print("H)")
mae = np.sum(np.abs(y-ypred))/n
print(f"MAE: {mae:.3f}")
rmse = np.sqrt(np.sum((y-ypred)**2)/n)
print(f"RMSE: {rmse:.3f}")
maex4 = np.sum(np.abs(y-Ypredx4))/n
print(f"MAE altr: {maex4:.3f}")
rmsex4 = np.sqrt(np.sum((y-Ypredx4)**2)/n)
print(f"RMSE altr: {rmsex4:.3f}")

#fig1 = px.scatter(x=x1, y=y, trendline="ols", labels={'x': 'Idade', 'y': 'Arsênio nas Unhas'},title="Relação entre Idade e Arsênio nas Unhas")
#fig2 = px.scatter(x=x2, y=y, trendline="ols", labels={'x': 'Uso_Beber', 'y': 'Arsênio nas Unhas'},title="Relação entre Uso para Beber e Arsênio nas Unhas")
#fig3 = px.scatter(x=x3, y=y, trendline="ols", labels={'x': 'Uso_Cozinhar', 'y': 'Arsênio nas Unhas'},title="Relação entre Uso para Cozinhar e Arsênio nas Unhas")
#fig4 = px.scatter(x=x4, y=y, trendline="ols", labels={'x': 'Arsenio_Agua', 'y': 'Arsênio nas Unhas'},title="Relação entre Arsenio na Água e Arsênio nas Unhas")

#fig1.show()
#fig2.show()
#fig3.show()
#fig4.show()
