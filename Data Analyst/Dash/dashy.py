import dash 
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input,Output
import pandas_datareader.data as web
import datetime

start=datetime.datetime(2018,1,1)
end=datetime.datetime.now()

app=dash.Dash()

app.layout= html.Div(children=[

    html.H1('Hello my first project'),
    dcc.Input(id='input',value='Type Some Thing here',type='text'),
    html.Div(id='graph'),
   
])#close div

@app.callback(
    Output(component_id='graph',component_property='children'),
    [Input(component_id='input',component_property='value')]
)


def update_graph(input_data):
    stock=web.DataReader(input_data,'morningstar',start,end)
    return dcc.Graph(
        id='stock',
        figure={
            'data':[
                {'x':stock.index,'y':stock.Close,'type':'line','name':input_data}
                    
            ],'layout':[{'title':input_data+' Stock'}]
            
        }#close figure
    )#close graph

if __name__ =='__main__':
    app.run_server(debug=True)