import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from pathlib import Path

st.set_page_config(page_title='Churn | Simulador financeiro', page_icon='📊', layout='wide')
st.title('📊 Churn em Telecom | Simulador financeiro')
st.caption('Validação dos modelos e cenários de retenção. Valores financeiros estimados, não resultados observados de uma campanha.')

@st.cache_data
def carregar_dados():
    caminho = Path(__file__).parent / 'dashboard_churn.csv'
    df = pd.read_csv(caminho, encoding='utf-8-sig')
    colunas = ['Churn_Real', 'MonthlyCharges', 'RF_Baseline', 'RF_Balanced', 'LogisticRegression']
    faltantes = [c for c in colunas if c not in df.columns]
    if faltantes:
        raise ValueError(f'Colunas ausentes no CSV: {faltantes}')
    df = df.copy()
    for c in ['Churn_Real', 'RF_Baseline', 'RF_Balanced', 'LogisticRegression']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
        if not df[c].isin([0, 1]).all():
            raise ValueError(f'A coluna {c} deve conter apenas 0 e 1, sem valores ausentes.')
        df[c] = df[c].astype(int)
    df['MonthlyCharges'] = pd.to_numeric(df['MonthlyCharges'], errors='coerce')
    if df['MonthlyCharges'].isna().any() or (df['MonthlyCharges'] < 0).any():
        raise ValueError('MonthlyCharges contém valores ausentes, inválidos ou negativos.')
    return df

try:
    dados = carregar_dados()
except Exception as e:
    st.error(f'Não foi possível carregar a base: {e}')
    st.stop()

st.sidebar.header('Premissas da campanha')
desconto = st.sidebar.slider('Desconto na mensalidade (%)', 0, 50, 10, 1) / 100
efetividade = st.sidebar.slider('Churners corretamente identificados que serão retidos (%)', 0, 100, 30, 5) / 100
margem = st.sidebar.slider('Margem de contribuição (%)', 0, 100, 70, 5) / 100
meses = st.sidebar.slider('Horizonte de retenção (meses)', 1, 24, 12, 1)
promocao = st.sidebar.radio('Duração da promoção (meses)', options=[1, 3, 5, 6], index=1, horizontal=True)
meses_desconto = min(promocao, meses)
st.sidebar.caption(f'O desconto incide por {meses_desconto} mês(es), dentro do horizonte de {meses} mês(es). Hipóteses editáveis, não estimativas causais.')

modelos = {
    'Random Forest baseline': 'RF_Baseline',
    'Random Forest balanceada': 'RF_Balanced',
    'Regressão logística': 'LogisticRegression',
}

def brl(v):
    return f'R$ {v:,.2f}'.replace(',', 'X').replace('.', ',').replace('X', '.')

def pct(v):
    return f'{100*v:.1f}%'.replace('.', ',')

st.subheader('Visão geral da amostra de validação')
k1, k2, k3, k4 = st.columns(4)
k1.metric('Clientes', f'{len(dados):,}'.replace(',', '.'))
k2.metric('Churn real', pct(dados['Churn_Real'].mean()))
k3.metric('Receita mensal da amostra', brl(dados['MonthlyCharges'].sum()))
k4.metric('Mensalidade média', brl(dados['MonthlyCharges'].mean()))

linhas = []
for nome, coluna in modelos.items():
    y = dados['Churn_Real']
    pred = dados[coluna]
    tp_mask = (y == 1) & (pred == 1)
    fp_mask = (y == 0) & (pred == 1)
    fn_mask = (y == 1) & (pred == 0)
    tn_mask = (y == 0) & (pred == 0)
    tp, fp, fn, tn = [int(mask.sum()) for mask in [tp_mask, fp_mask, fn_mask, tn_mask]]
    receita_tp = dados.loc[tp_mask, 'MonthlyCharges'].sum()
    receita_fp = dados.loc[fp_mask, 'MonthlyCharges'].sum()
    # Benefício = contribuição dos churners que a campanha efetivamente retém.
    beneficio = receita_tp * efetividade * margem * meses
    # Desconto para falsos positivos e para churners efetivamente retidos.
    custo_fp = receita_fp * desconto * meses_desconto
    custo_tp = receita_tp * efetividade * desconto * meses_desconto
    custo = custo_fp + custo_tp
    liquido = beneficio - custo
    roi = liquido / custo if custo > 0 else np.nan
    denominador_equilibrio = receita_tp * (margem * meses - desconto * meses_desconto)
    breakeven = (receita_fp * desconto * meses_desconto / denominador_equilibrio
                 if denominador_equilibrio > 0 else np.nan)
    linhas.append({
        'Modelo': nome, 'Clientes abordados': tp + fp, 'VP': tp, 'FP': fp, 'FN': fn, 'VN': tn,
        'Precisão': tp / (tp + fp) if tp + fp else 0,
        'Recall': tp / (tp + fn) if tp + fn else 0,
        'F1': 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0,
        'Acurácia': (tp + tn) / len(dados),
        'Benefício': beneficio, 'Desconto FP': custo_fp, 'Desconto VP retidos': custo_tp,
        'Custo total': custo, 'Resultado líquido': liquido, 'ROI': roi,
        'Efetividade de equilíbrio': breakeven,
    })
resultado = pd.DataFrame(linhas)

st.subheader('Comparação financeira entre modelos')
st.caption(f'Cenário: {pct(desconto)} de desconto por {meses_desconto} mês(es), horizonte de {meses} mês(es), efetividade {pct(efetividade)} e margem {pct(margem)}.')
colunas_tabela = ['Modelo', 'Clientes abordados', 'VP', 'FP', 'Precisão', 'Recall', 'F1', 'Benefício', 'Custo total', 'Resultado líquido', 'ROI']
tabela = resultado[colunas_tabela].copy()
for c in ['Precisão', 'Recall', 'F1', 'ROI']:
    tabela[c] = tabela[c].map(lambda x: '—' if pd.isna(x) else pct(x))
for c in ['Benefício', 'Custo total', 'Resultado líquido']:
    tabela[c] = tabela[c].map(brl)
st.dataframe(tabela, use_container_width=True, hide_index=True)

fig = px.bar(resultado, x='Modelo', y='Resultado líquido', text='Resultado líquido',
             title='Resultado financeiro líquido por modelo', labels={'Resultado líquido': 'R$'})
fig.update_traces(texttemplate='R$ %{y:,.0f}', textposition='outside')
st.plotly_chart(fig, use_container_width=True)

escolhido = st.selectbox('Detalhar modelo', list(modelos), index=1)
r = resultado.loc[resultado['Modelo'] == escolhido].iloc[0]
st.subheader(f'Detalhamento: {escolhido}')
a, b, c = st.columns(3)
a.metric('Benefício estimado', brl(r['Benefício']))
b.metric('Custo dos descontos', brl(r['Custo total']))
c.metric('Resultado líquido', brl(r['Resultado líquido']))

detalhe = pd.DataFrame({
    'Componente': ['Benefício da retenção', 'Desconto para falsos positivos', 'Desconto para churners retidos'],
    'Valor': [r['Benefício'], -r['Desconto FP'], -r['Desconto VP retidos']],
})
fig2 = px.bar(detalhe, x='Componente', y='Valor', color='Componente',
              title='Componentes do resultado financeiro (R$)')
fig2.update_layout(showlegend=False)
st.plotly_chart(fig2, use_container_width=True)

if pd.notna(r['Efetividade de equilíbrio']):
    st.info(f"Efetividade mínima estimada para equilíbrio financeiro: **{pct(r['Efetividade de equilíbrio'])}**. "
            'Valores superiores a 100% indicam que as premissas atuais não permitem atingir o equilíbrio.')
else:
    st.info('Não há ponto de equilíbrio positivo calculável com as premissas atuais: a contribuição preservada por VP não supera seu próprio desconto, ou não há receita de VP.')

st.subheader('Sensibilidade à efetividade da campanha')
st.caption('Resultado líquido estimado para cada modelo, variando a taxa de retenção efetiva de 0% a 100%.')
curvas = []
for nome, coluna in modelos.items():
    pred = dados[coluna]
    receita_tp = dados.loc[(dados['Churn_Real'] == 1) & (pred == 1), 'MonthlyCharges'].sum()
    receita_fp = dados.loc[(dados['Churn_Real'] == 0) & (pred == 1), 'MonthlyCharges'].sum()
    for taxa in np.linspace(0, 1, 101):
        liquido_sensibilidade = (receita_tp * taxa * margem * meses
                               - receita_fp * desconto * meses_desconto
                               - receita_tp * taxa * desconto * meses_desconto)
        curvas.append({'Modelo': nome, 'Efetividade (%)': taxa * 100,
                       'Resultado líquido (R$)': liquido_sensibilidade})
fig3 = px.line(pd.DataFrame(curvas), x='Efetividade (%)', y='Resultado líquido (R$)',
               color='Modelo', title='Sensibilidade do resultado financeiro')
fig3.add_hline(y=0, line_dash='dash', line_color='gray')
st.plotly_chart(fig3, use_container_width=True)


with st.expander('Metodologia e limitações'):
    st.markdown('''**VP**: cliente que realmente cancelou e foi identificado pelo modelo. **FP**: cliente que não cancelou, mas foi identificado para oferta.

- **Benefício** = soma das mensalidades dos VP × efetividade × margem × meses.
- **Custo FP** = soma das mensalidades dos FP × desconto × meses de promoção (limitados ao horizonte).
- **Custo VP retidos** = soma das mensalidades dos VP × efetividade × desconto × meses de promoção (limitados ao horizonte).
- **Resultado líquido** = benefício − custo FP − custo VP retidos.
- **ROI** = resultado líquido ÷ custo total; não é definido quando o custo é zero.

O ponto de equilíbrio considera benefício ao longo do horizonte e desconto apenas durante a promoção. O cálculo supõe que clientes retidos permaneçam durante todo o horizonte; clientes que recusam a oferta não geram receita nem desconto; todos os FP abordados recebem desconto; não há custo operacional, impostos ou custo de aquisição. O desconto reduz receita, mas o cálculo não modela mudanças de consumo. A efetividade é uma hipótese e não foi identificada por experimento causal. Métricas dos modelos são da mesma amostra de validação, que também foi usada para escolher limiares; resultados podem ser otimistas. A avaliação financeira é um cenário, não uma previsão garantida.''')
