from dash import Dash, dcc, html, Input, Output, callback, State, dash_table, no_update, callback_context
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import requests
import threading
import queue
import os
import shutil
from datetime import datetime, timedelta

import plotly.graph_objects as go
import json
import openai
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import PyPDFLoader


# TODO: 'chroma.sqlite3' cant be deleted while vectordb is not set to None
# Setting vectordb to None will require creating new Chroma() instance to query/access db again that will tax embedding model
os.environ["OPENAI_API_KEY"] = ""
thread = None
pdf_dates = {}
current_date = datetime.now()
# NOTE: plot_gpt() can return pandas df from json response, you can use df for datatable output
fig = go.Figure()
persist_directory = 'db'
output_directory = "downloaded_pdfs"
os.makedirs(output_directory, exist_ok=True)
# HACK: Mutable dict flag to store data, instead of global var
vectordb_dict = {'vectordb': None}
docs_dict = {'docs_list': None}
plot_dict = {'title': None,'query': None}
# HACK: Queue to communicate between the downloads thread and Dash app
progress_queue = queue.Queue()
# HACK: flag to control the threads execution
stop_thread_flag = threading.Event()
database_flag = threading.Event()
seconddb_flag = threading.Event()

# Initialize the Dash app
app = Dash(__name__, external_stylesheets=[dbc.themes.FLATLY, dbc.icons.BOOTSTRAP])
server = app.server

password_input = dbc.Input(
    id='password-input',
    type='password',
    placeholder=['Enter your key...'],
    value=None
)

button_group = html.Div(
    [
        dbc.Row([
                dbc.RadioItems(
                    id="radios",
                    className="btn-group mt-3",
                    inputClassName="btn-check",
                    labelClassName="btn btn-outline-primary",
                    labelCheckedClassName="active",
                    options=[
                        {"label": "CPI Inflation", "value": "CPI"},
                        {"label": "Interest Rates", "value": "Rates"},
                        {"label": "GDP Growth", "value": "GDP"},
                        {"label": "Consumer Spending", "value": "Consumption"},
                        {"label": "Household Savings", "value": "Savings"},
                        {"label": "Unemployment Rate", "value": "Unemployment"},
                        {"label": "Oil Prices", "value": "Oil"}
                    ], value="")
        ]),
        html.Div(id="output"),
        dbc.Card(
            dbc.Row([
                dbc.InputGroup([
                    dbc.Textarea(id='prompt-input',placeholder="Type something..."),
                    dbc.Button(children=["Submit Prompt ", html.I(className="bi bi-send", style=dict(display='inline-block',fontSize='1rem'))], id='submit-button', n_clicks=0)
                    ]),
                html.Div(id="output-prompt")
            ]), body=True, className="shadow-sm p-3 mt-3 bg-light rounded"
        )
    ],
    className="radio-group",
    id="button-group" 
)

collapse = html.Div(
    [
        html.Br(),
        dbc.Button(
            children=[html.I(className="bi bi-graph-up-arrow", style=dict(display='inline-block')),' Generate a Plot'],
            id="collapse-button",
            outline=False,
            size="lg",
            className="mb-3",
            color="primary",
            n_clicks=0,
        ),
        dbc.Collapse(
            dbc.Card(dcc.Graph(figure=fig,id='plot-device')),
            id="collapse",
            is_open=False,
        )
    ]
)

def dummy_download_and_progress():
    return html.Div([
        html.Div(id='dummy-output', style=dict(display='none')),
        dcc.Interval(id='interval-component', interval=500, n_intervals=0, disabled=False),
        dbc.Button(children=["1. Press to Download Latest Reports ", html.I(className="bi bi-file-earmark-text", style=dict(display='inline-block'))],
                   id="download-button", color="info", className='mr-3 mt-3 mb-3',n_clicks=0, outline=True),
        dbc.Button(children=["2. Press to Process Reports into a Database ", html.I(className="bi bi-database-add", style=dict(display='inline-block'))], 
                   id="chunking-button", color="info", className='mr-3 mt-3 mb-3 d-none',n_clicks=0, outline=True),
        dbc.Progress(id="progress-bar", value=0, label="",style=dict(height='20px'),className='d-none'),
        dash_table.DataTable(
            id='pdf-dates-table',
            columns=[{"name": i, "id": i} for i in ['File Path', 'Author', 'Date']],
            data=[], style_table={'display': 'none', 'margin-bottom': '3px'} 
        ),
        html.Div([
            html.I(className="bi bi-book", style=dict(display='inline-block',fontSize='2rem')),
            html.H3('Select a Topic', className='mr-3 mt-0 mb-0', style={'fontWeight': 'bold','display':'inline-block','margin-left':'10px'})
        ]),
        button_group,
        collapse,
        dbc.Row([
        dbc.Col([
            html.Footer([
                html.P("Developed by Victor Zommers | ",
                       style={'display': 'inline-block','font-size': '16px'}),
                html.A("Check out other dashboards", href="https://sites.google.com/view/victor-zommers/",
                       style={'display': 'inline-block', 'margin-left': '5px','font-size': '16px'},target="_blank"),
                html.Span(" | ", style={'display': 'inline-block','font-size': '16px', 'margin-left': '5px'}),
                html.A("Get in touch", href="mailto:vic.dashboards@icloud.com?subject=[GitHub]%20LLM%20Dashboard",
                       style={'display': 'inline-block', 'margin-left': '5px','font-size': '16px'},target="_blank")
            ], style={'text-align': 'left', 'margin-top': '10px'})
        ], width=12)
        ])
    ])

def validate_password(password):
    if password is None:
        return dbc.Alert("No Key yet", color='light')
    elif not str(password).startswith("sk-") or len(password) <7:
        return dbc.Alert("hmm... Key is invalid", color='secondary')
    os.environ["OPENAI_API_KEY"] = str(password)
    #'*'*(len(password)-6 #alternative
    return dbc.Alert('Your key: '+password[:3] + ' .'*3 + password[-4:],color='success')

app.title = 'Central Bank Speak'
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.Img(src='assets/logo.png', style={'width':'100%', 'margin-top': '10px'}),width=3),
        dbc.Col([
            html.Div([
                html.I(className="bi bi-bank", style=dict(display='inline-block',fontSize='5rem')),
                html.H2('Central Bank Speak (RAG-LLM)', style={'fontWeight': 'bold','display':'inline-block','margin-left':'10px'}),
            ]),
            html.P('Summarise or Plot latest Macroeconomic projections and Inflation forecasts from Central Bank publications using ChatGPT & Langchain.'),
            html.Div([
                html.I(className="bi bi-question-square", style=dict(display='inline-block',fontSize='2rem')),
                html.H3('How to use:', style={'fontWeight': 'bold','display':'inline-block','margin-left':'10px'})
            ]),
            html.Ol([
                    html.Li('Paste your unique OpenAI API key below'),
                    html.Li('Select a currency/country tab'),
                    html.Li('Download latest reports, Process them into a Database'),
                    html.Li('Select a topic to query'),
                    html.Li('Submit a prompt to ChatGPT or Generate a plot')
                ], style={'list-style-type': 'upper-roman', 'list-style-position': 'inside', 'text-align': 'left', 'padding-left': '0'})
        ],width=9)
    ]),
    dbc.Row([
        html.Hr(),
        dbc.Row([
            dbc.Col(dbc.InputGroup([
                dbc.InputGroupText([html.I(className="bi bi-key", style=dict(fontSize='2rem', display='inline-block')),
                                    html.H5(['OpenAI API Key',html.Span('*', style={'color': 'red','fontWeight': 'bold'})], style={'display':'inline-block','margin-left':'10px'})
                                    ], style={'display': 'flex', 'justify-content': 'center', 'align-items': 'center'}),
                password_input
            ]), width=8),
            dbc.Col([html.Div(id='password-output')],width=4)
        ]),
        html.P(['Please add your OpenAI API key. It will be used to generate your summary and visualisations from downloaded reports and your custom prompt. You can set-up your key ',
                html.A([html.Span('HERE', style={'textDecoration': 'underline'}), html.Span(' '), 
                        html.I(className="bi bi-box-arrow-up-right", style=dict(display='inline-block',fontSize='1rem'))], 
                       href='https://platform.openai.com/account/api-keys', target='_blank',style={'fontWeight': 'bold', 'textDecoration': 'none'})
                ],className='text-primary'),
        html.Hr()
    ]),
    dcc.Tabs(id="tabs", value='tab-2', children=[
        dcc.Tab(label='US Dollar $', value='tab-1', selected_className='fw-bold', selected_style={'background-color': 'green', 'color': 'white','border-top':'1px  darkgrey solid','opacity':'0.75'}),
        dcc.Tab(label='Pound Stirling £', value='tab-2', selected_className='fw-bold', selected_style={'background-color': 'red', 'color': 'white','border-top':'1px  darkgrey solid','opacity':'0.75'}),
        dcc.Tab(label='Euro Area €', value='tab-3', selected_className='fw-bold', selected_style={'background-color': 'darkblue', 'color': 'white','border-top':'1px  darkgrey solid','opacity':'0.50'})
    ]),
    html.Div(id='tabs-content'),
], fluid=False)

def boe_download():
    download_progress = 0
    pdf_dates.clear()
    os.makedirs(output_directory, exist_ok=True)
    base_url = "https://www.bankofengland.co.uk/-/media/boe/files/monetary-policy-report/"
    for i in range(12):
        if stop_thread_flag.is_set():
            print("Stopping as requested")
            download_progress = 0
            progress_queue.put(download_progress)
            return
        # Calculate the date for the current iteration
        past_date = current_date - timedelta(days=i * 30)  # Assuming 30 days per month for simplicity
        # Format the date components (month and year)
        month_str = past_date.strftime("%B").lower()
        year_str = str(past_date.year)
        # Construct the URL
        pdf_url = f"{base_url}{year_str}/{month_str}/monetary-policy-report-{month_str}-{year_str}.pdf"
        # Define the local file path
        month_year = pdf_url.split("/")[-2:]
        local_file_path = os.path.join(output_directory, f"monetary-policy-report-{month_year[0]}-{month_year[1]}.pdf")
        # Download and save the PDF file locally
        response = requests.get(pdf_url)
        if response.status_code == 200:
            with open(local_file_path, "wb") as pdf_file:
                pdf_file.write(response.content)
            print(f"Downloaded: {pdf_url}")
            pdf_dates[local_file_path] = {
                'date': datetime(year=past_date.year, month=past_date.month, day=1).strftime('%Y-%m-%dT%H:%M:%S%z'),
                'author': 'Bank of England'}
        download_progress += 8.33
        progress_queue.put(download_progress)

def fed_download():
    download_progress = 0
    pdf_dates.clear()
    os.makedirs(output_directory, exist_ok=True)
    base_url = "https://www.federalreserve.gov/monetarypolicy/files/"
    # Loop through all dates in the past 360 days
    for k in range(365):
        if stop_thread_flag.is_set():
            print("Stopping as requested")
            download_progress = 0
            progress_queue.put(download_progress)
            return
        # Calculate the date for the current iteration
        target_date = current_date - timedelta(days=k)
        # Format the date components (year, month, and day)
        year_str = target_date.strftime("%Y")
        month_str = target_date.strftime("%m")
        day_str = target_date.strftime("%d")
        # Construct the URL for the Fed release
        pdf_url = f"{base_url}fomcminutes{year_str}{month_str}{day_str}.pdf"
        # Define the local file path
        local_file_path = os.path.join(output_directory, f"fomcminutes{year_str}{month_str}{day_str}.pdf")
        # Download the PDF file
        response = requests.get(pdf_url)
        if response.status_code == 200:
            with open(local_file_path, "wb") as pdf_file:
                pdf_file.write(response.content)
            pdf_url2 = f"{base_url}monetary{year_str}{month_str}{day_str}a1.pdf"
            local_file_path2 = os.path.join(output_directory, f"monetary{year_str}{month_str}{day_str}a1.pdf")
            with open(local_file_path2, "wb") as pdf_file2:
                pdf_file2.write(requests.get(pdf_url2).content)
            print(f"Downloaded: {pdf_url}")
            print(f"Downloaded: {pdf_url2}")
            pdf_dates[local_file_path] = {
                'date': datetime(year=target_date.year, month=target_date.month, day=1).strftime('%Y-%m-%dT%H:%M:%S%z'),
                'author': 'FOMC Minutes'}
            pdf_dates[local_file_path2] = {
                'date': datetime(year=target_date.year, month=target_date.month, day=1).strftime('%Y-%m-%dT%H:%M:%S%z'),
                'author': 'FOMC Statement'}
        download_progress = k/3.65
        progress_queue.put(download_progress)

def ecb_download():
    download_progress = 0
    pdf_dates.clear()
    os.makedirs(output_directory, exist_ok=True)
    # Define the base URL pattern
    base_url = "https://www.ecb.europa.eu/pub/pdf/ecbu/"
    # Calculate the release dates for the latest 8 releases
    release_dates = []
    for j in range(8):
        release_date = current_date - timedelta(days=j * 365 // 8)
        release_dates.append(release_date)
    # Iterate over the release dates and generate URLs
    for release_date in release_dates:
        if stop_thread_flag.is_set():
            print("Stopping as requested")
            download_progress = 0
            progress_queue.put(download_progress)
            return
        year_str = str(release_date.year)
        # Calculate the release name based on the day of the year
        day_of_year = release_date.timetuple().tm_yday
        if day_of_year > 335 or day_of_year <30: release_name = 8
        else: release_name = ((day_of_year - 1) // 45) + 1
        release_name_str = f"{release_name:02d}"
        pdf_url = f"{base_url}eb{year_str}{release_name_str}.en.pdf"
        # Define the local file path
        local_file_path = os.path.join(output_directory, f"eb{year_str}{release_name_str}.en.pdf")
        # Download and save the PDF file locally
        response = requests.get(pdf_url)
        if response.status_code == 200:
            with open(local_file_path, "wb") as pdf_file:
                pdf_file.write(response.content)
            print(f"Downloaded: {pdf_url}")
            pdf_dates[local_file_path] = {
                'date': datetime(year=release_date.year, month=release_date.month, day=1).strftime('%Y-%m-%dT%H:%M:%S%z'),
                'author': 'European Central Bank'}
        download_progress += 12.5
        progress_queue.put(download_progress)

def pipe_langchain(folder):
    database_flag.set()
    loader = DirectoryLoader(output_directory, glob="./*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    for doc in documents:
        if doc.metadata['source'] in pdf_dates:
            doc.metadata['date'] = pdf_dates[doc.metadata['source']]['date']
            doc.metadata['author'] = pdf_dates[doc.metadata['source']]['author']
    # splitting the text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    embedding = OpenAIEmbeddings()
    # Supplying a persist_directory will store the embeddings on disk
    vectordb_dict['vectordb'] = Chroma.from_documents(documents=texts,
                                    embedding=embedding,
                                    persist_directory=folder)
    # persist the db to disk
    vectordb_dict['vectordb'].persist()
    #vectordb_dict['vectordb'] = None
    #vectordb_dict['vectordb'] = Chroma(persist_directory=folder,
    #                    embedding_function=embedding)
    # HACK: we set flag when vectordb was persisted for interval input to listen
    database_flag.clear()
    seconddb_flag.set()
    print("Done with vectorising")
       
def plot_gpt(plot_prompt, plot_title):
    if docs_dict['docs_list'] is not None and seconddb_flag.is_set():
        openai.api_key = os.environ.get("OPENAI_API_KEY")
        chunk1 = docs_dict['docs_list'][0]
        chunk2 = docs_dict['docs_list'][1]
        chunk3 = docs_dict['docs_list'][2]
        chunk4 = docs_dict['docs_list'][3]
        chunk5 = docs_dict['docs_list'][4]
        system_prompt = f"""
        You are expert macroeconomic analyst. Your purpose is to find numeric information like trends, forecasts, predictions, estimates from central banks publication supplied and return it in a json format only!
        json should always have 2 keys: Date in strictly YYYY-MM-DD format, Value in numeric float. here is an example of acceptable json output (you can have as many rows as you want):
        {{
        "2023-01-01": 123.45,
        "2023-02-01": 234.56,
        "2023-03-01": 345.67
        }}
        Below are paragraphs delimited by ---.
        ---
        {chunk1}
        ---
        {chunk2}
        ---
        {chunk3}
        ---
        {chunk4}
        ---
        {chunk5}
        ---
        Each paragraph has its own publication date, if facts conflict, use paragraphs with latest publication date. Please answer the following user question (strictly) based on the facts above, you can do it!:
        """
        context = [{"role": "system", "content": system_prompt},{"role": "user", "content": plot_prompt}]
        # Send a message and receive a response
        response = openai.ChatCompletion.create(
            model="gpt-4-1106-preview",
            messages=context
        )
        # Extract and print the assistant's reply
        assistant_reply = response.choices[0].message['content']
        start = assistant_reply.find("{")
        end = assistant_reply.find("}") + 1
        json_data = assistant_reply[start:end]
        # Parse the JSON data into a Python dictionary
        data_dict = json.loads(json_data)
        # NOTE Pandas method if needed
        #df_plot = pd.DataFrame(list(data_dict.items()), columns=['Date', 'Value'])
        #fig = go.Figure(data=go.Scatter(x=df_plot['Date'], y=df_plot['Value'], mode='lines+markers', marker=dict(size=8), name='Scatter Plot'))
        #fig.update_layout(title=f"Time-Series Plot of {str(plot_title)}", xaxis_title='Date', yaxis_title=str(plot_title), template="simple_white")
        x_values = list(data_dict.keys())
        y_values = list(data_dict.values())
        fig = go.Figure(data=go.Scatter(x=x_values, y=y_values, mode='lines+markers', marker=dict(size=8), name='Scatter Plot'))
        fig.update_layout(title=f"Time-Series Plot of {str(plot_title)}", xaxis_title='Date', yaxis_title=str(plot_title), template="simple_white")
        return fig

# NOTE: Callback to initiate download and update styles of buttons depending on progress
@app.callback(
    #Output('progress-bar', 'value', allow_duplicate=True),
    Output('progress-bar', 'className'),
    Output('pdf-dates-table', 'style_table'),
    Output('chunking-button', 'className', allow_duplicate=True),
    Output('download-button', 'className'), 
    Output('download-button', 'disabled'),
    Input('download-button', 'n_clicks'), 
    Input('progress-bar', 'value'),
    Input('tabs', 'value'),
    prevent_initial_call=True
)
def control_processing(stop_clicks, progress, tab_val):
    changed_id = [p['prop_id'] for p in callback_context.triggered][0]
    if 'download-button' in changed_id:
        # NOTE: Delete old files, Reset the stop flag and start the thread but only when Download is pressed
        stop_thread_flag.clear()
        if os.path.exists(output_directory):
            shutil.rmtree(output_directory)
        if os.path.exists(persist_directory):
            shutil.rmtree(persist_directory, ignore_errors=True)
        if tab_val == 'tab-1':
            threading.Thread(target=fed_download).start()
        elif tab_val == 'tab-2':
            threading.Thread(target=boe_download).start()
        elif tab_val == 'tab-3':
            threading.Thread(target=ecb_download).start()
        return 'mr-3 mt-3 mb-3', {'display': 'block', 'margin-bottom': '3px'}, no_update, no_update, True
    elif 'tabs' in changed_id:
        # NOTE: logic moved to render_content()
        # Set the stop flag
        # stop_thread_flag.set()
        return 'd-none', {'display': 'none'}, no_update, no_update, False
    elif 'progress-bar' in changed_id and progress > 98:
        return 'd-none', {'display': 'none'}, 'mr-3 mt-3 mb-3', 'd-none', no_update
    return no_update, no_update, no_update, no_update, no_update  # Reset start button clicks to prevent multiple starts


# NOTE: Callback to update download progress bar
# It pretty much can be left on its own
# But we need to add datatable update & anything else that might require interval
@app.callback(
    Output('progress-bar', 'value'),
    Output('progress-bar', 'label'),
    Output('pdf-dates-table', 'data'),
    Output('chunking-button', 'className'),
    Input('interval-component', 'n_intervals')
)
def update_progress(n):

    while not progress_queue.empty():
        try:
            # Check if there's a new item in the queue
            item_val = progress_queue.get_nowait()
            progress = round(item_val,1)
            label = f'{progress}% (Downloading PDFs)'
            data = [{'File Path': key, 'Author': pdf_dates[key]['author'], 'Date': pdf_dates[key]['date']} for key in pdf_dates.keys()]
            return progress, label, data, 'd-none'
        except queue.Empty:
            break  
    if seconddb_flag.is_set() and vectordb_dict['vectordb'] is not None:
        return no_update, no_update, no_update, 'd-none'
    return no_update, no_update, no_update, no_update  # If no new progress or data, return no update

@callback(Output('tabs-content', 'children'),
          Output('password-output', 'children'),
          Input('tabs', 'value'),
          Input('password-input', 'value'))

def render_content(tab, password):
    # HACK: this should prevent switching tab while pipe_langchain() is running, i.e. database is being set
    # seconddb_flag is needed to keep chunking button enabled when tab is switched but vectordb is not None
    if database_flag.is_set():
        raise PreventUpdate
    if tab == 'tab-1': # we need a flag for DB_run
        docs_dict['docs_list'] = None
        stop_thread_flag.set()
        seconddb_flag.clear()
        return html.Div([
            html.I(className="bi bi-bank2", style=dict(display='inline-block',fontSize='2rem')),
            html.H3('Federal Reserve', className='mr-3 mt-3 mb-0', style={'display':'inline-block','margin-left':'10px'}),
            dummy_download_and_progress()
        ]), validate_password(password)
    elif tab == 'tab-2':
        docs_dict['docs_list'] = None
        stop_thread_flag.set()
        seconddb_flag.clear()
        return html.Div([
            html.I(className="bi bi-bank2", style=dict(display='inline-block',fontSize='2rem')),
            html.H3('Bank of England', className='mr-3 mt-3 mb-0', style={'display':'inline-block','margin-left':'10px'}),
            dummy_download_and_progress()
        ]), validate_password(password)
    elif tab == 'tab-3':
        docs_dict['docs_list'] = None
        stop_thread_flag.set()
        seconddb_flag.clear()
        return html.Div([
            html.I(className="bi bi-bank2", style=dict(display='inline-block',fontSize='2rem')),
            html.H3('European Central Bank', className='mr-3 mt-3 mb-0', style={'display':'inline-block','margin-left':'10px'}),
            dummy_download_and_progress()
        ]), validate_password(password)

@app.callback(
    Output('chunking-button', 'children'),
    Output('chunking-button', 'disabled'),
    Input('chunking-button', 'n_clicks'),
    prevent_initial_call=True
)

def chuncking(n_clicks):
    if n_clicks > 0:
        if os.environ["OPENAI_API_KEY"].startswith("sk-"):
            thread = threading.Thread(target=pipe_langchain, args=(persist_directory,))
            thread.start()
            return [dbc.Spinner(size="sm"), " Setting-up a Vector Database..."], True
        else:
            return ["2. SUPPLY A VALID KEY TO PROGRESS ",html.I(className="bi bi-exclamation-triangle", style=dict(display='inline-block'))], True

# callback to print docs[0] in "output" of button_group when dbc.RadioItems is selected
@app.callback(
    Output("output", "children"),
    Input("radios", "value"),
    State("tabs", "value")
)

def print_docs(radio_val,tab_val):
    docs_dict['docs_list'] = None
    query = ''
    if tab_val == 'tab-1':
        if radio_val == 'CPI':
            query = f"forecast outlook for CPI Inflation for {current_date.year} and next 5 years"
        elif radio_val == 'Rates':
            query = f"forecast outlook and past FOMC decisions for federal funds rate {current_date.year} and next 5 years"
        elif radio_val == 'GDP':
            query = f"forecast outlook for economic activity and Real GDP Growth for {current_date.year} and next 5 years"
        elif radio_val == 'Consumption':
            query = f"forecast outlook for Consumer Spending and Retail Sales for {current_date.year} and next 5 years"
        elif radio_val == 'Savings':
            query = f"forecast outlook for Household Savings for {current_date.year} and next 5 years"
        elif radio_val == 'Unemployment':
            query = f"forecast outlook for Labor Market and Unemployment Rate for {current_date.year} and next 5 years"
        elif radio_val == 'Oil':
            query = f"forecast outlook for Oil Prices and Energy for {current_date.year} and next 5 years"
    elif tab_val == 'tab-2':
        if radio_val == 'CPI':
            query = f"MPC forecast outlook for CPI Inflation for {current_date.year} and next 5 years"
        elif radio_val == 'Rates':
            query = f"MPC forecast outlook and past decisions for Bank Rate for {current_date.year} and next 5 years"
        elif radio_val == 'GDP':
            query = f"MPC forecast and economic outlook for Real GDP Growth in the UK for {current_date.year} and next 5 years"
        elif radio_val == 'Consumption':
            query = f"MPC forecast outlook for Consumer Spendings for {current_date.year} and next 5 years"
        elif radio_val == 'Savings':
            query = f"MPC forecast outlook for Household Saving Ratio for {current_date.year} and next 5 years"
        elif radio_val == 'Unemployment':
            query = f"MPC forecast outlook for Labour Market and Unemployment Rate in the UK for {current_date.year} and next 5 years"
        elif radio_val == 'Oil':
            query = f"MPC forecast outlook for Oil and Energy Prices for {current_date.year} and next 5 years"
    elif tab_val =='tab-3':
        if radio_val == 'CPI':
            query = f"forecast outlook for CPI Inflation in Euro area for {current_date.year} and next 5 years"
        elif radio_val == 'Rates':
            query = f"past decisions of Governing Council and future outlook for key ECB interest rates (MRO, deposit facility, marginal lending facility) for {current_date.year} and next 5 years"
        elif radio_val == 'GDP':
            query = f"forecast outlook for economic activity and Real GDP Growth in Euro area for {current_date.year} and next 5 years"
        elif radio_val == 'Consumption':
            query = f"forecast outlook for Consumer Spendings and Retail Sales in Euro area for {current_date.year} and next 5 years"
        elif radio_val == 'Savings':
            query = f"forecast outlook for Households Saving Rate for {current_date.year} and next 5 years"
        elif radio_val == 'Unemployment':
            query = f"forecast outlook for Labour Market and Unemployment in Euro Area for {current_date.year} and next 5 years"
        elif radio_val == 'Oil':
            query = f"forecast outlook for Oil and Energy Prices for {current_date.year} and next 5 years"
    plot_dict["title"] = radio_val
    plot_dict["query"] = query
    if database_flag.is_set():
        return dbc.Alert("the Database is being set-up... Please wait for the spinner to disappear and retry your selection!", color="warning", className='mt-3')
    elif stop_thread_flag.is_set():
        # NOTE: This branch deletes/tries vectordb when tab is swithced. First, it listens to thread 
        if vectordb_dict['vectordb'] is not None:
            if thread is not None:
                thread.join()
            print('pipelang closed')
            vectordb_dict['vectordb'].delete_collection()
            vectordb_dict['vectordb'].persist()
            vectordb_dict['vectordb'] = None
            # FIXME: need to format ./db folder, otherwise new documents are appended to old ones, I keep chroma in-memory for now
            # TODO: delete db/chroma.sqlite3
        return html.Div()
    elif vectordb_dict['vectordb'] is not None and not database_flag.is_set():
        # NOTE: This branch does heavy lifting of retreving relevant chunks from vectordb
        retriever = vectordb_dict['vectordb'].as_retriever(search_kwargs={"k": 8})
        docs_dict['docs_list'] = retriever.get_relevant_documents(query, verbose=True)
        if docs_dict['docs_list'] is not None:
            print(f"query:{query}")
            print(f"Doc:{docs_dict['docs_list'][0]}")
        metadata_copy = docs_dict['docs_list'][0].metadata.copy()
        metadata_copy['date'] = datetime.strptime(metadata_copy['date'], '%Y-%m-%dT%H:%M:%S').strftime('%Y-%b')
        metadata_copy['source'] = metadata_copy['source'].split(os.sep)[-1][:-4]
        # FIXME: retriever prevents db/chroma.sqlite3 from being deleted
        # TODO: assign vectirdb to None to close connection, NOTE: connection cant be reinstated without rerunning embedding model
        return dbc.Card([
            html.H5('Most Relevant Context:',style={'fontWeight': 'bold'}),
            dcc.Markdown(docs_dict['docs_list'][0].page_content,dangerously_allow_html=True),
            dcc.Markdown("\n\n".join([f'{key}: {value}' for key, value in metadata_copy.items()]), dangerously_allow_html=True)
            ], body=True, className="shadow-sm p-3 mt-3 bg-light rounded")
    else: 
        return html.Div()

@app.callback(
    Output('submit-button', 'children'),
    Output('submit-button', 'disabled'),
    Input('submit-button', 'n_clicks'),
    prevent_initial_call=True
)

def submit_func(n_clicks):
    if n_clicks > 0 and seconddb_flag.is_set():
        return [dbc.Spinner(size="sm"), " Thinking..."], True

@app.callback(
    Output("output-prompt", "children"),
    Output('submit-button', 'children', allow_duplicate=True),
    Output('submit-button', 'disabled', allow_duplicate=True),
    [Input('submit-button', 'n_clicks')],
    [State("prompt-input", "value")],
    prevent_initial_call=True
)

def prompt_result(n_clicks,texty):
    if n_clicks > 0:
        if docs_dict['docs_list'] is not None:
            openai.api_key = os.environ.get("OPENAI_API_KEY")
            chunk1 = docs_dict['docs_list'][0]
            chunk2 = docs_dict['docs_list'][1]
            chunk3 = docs_dict['docs_list'][2]
            chunk4 = docs_dict['docs_list'][3]
            chunk5 = docs_dict['docs_list'][4]
            chunk6 = docs_dict['docs_list'][5]
            chunk7 = docs_dict['docs_list'][6]
            chunk8 = docs_dict['docs_list'][7]
            system_prompt = f"""
            You are expert macroeconomic analyst. Your purpose is to summarise and find relevant information such as trends, forecasts, judgements, reasons from central banks publication supplied.
            Below are paragraphs delimited by ---. Return metadata of paragraphs used in answer at the end of response (source, page, date), always split metadata from the rest of response by ///.
            ---
            {chunk1}
            ---
            {chunk2}
            ---
            {chunk3}
            ---
            {chunk4}
            ---
            {chunk5}
            ---
            {chunk6}
            ---
            {chunk7}
            ---
            {chunk8}
            ---
            Each paragraph has its own publication date, if facts conflict, use paragraphs with latest publication date. Please answer the following user question (strictly) based on the facts above, you can do it!:
            """
            context = [

                {"role": "system", "content": system_prompt},
                {"role": "user", "content": texty}
            ]
            # Send a message and receive a response
            response = openai.ChatCompletion.create(
                model="gpt-4-1106-preview",
                messages=context
            )
            # Extract and print the assistant's reply
            assistant_reply = response.choices[0].message['content']
            return html.Div([html.Br(), dcc.Markdown(assistant_reply,dangerously_allow_html=True)]), ['Submit Prompt ',html.I(className="bi bi-send", style=dict(display='inline-block',fontSize='1rem'))], False
        return no_update, no_update  , no_update    

@app.callback(
    Output("collapse-button", "children", allow_duplicate=True),
    Output("collapse-button", "disabled", allow_duplicate=True),
    [Input("collapse-button", "n_clicks")],
    prevent_initial_call=True
)

def collapse_loading(n):
    if n > 0:
        return [dbc.Spinner(size="sm"), " Thinking..."], True

@app.callback(
    Output("collapse", "is_open"),
    Output("plot-device", "figure"),
    Output("collapse-button", "children"),
    Output("collapse-button", "disabled"),
    [Input("collapse-button", "n_clicks")],
    [State("collapse", "is_open")]
)

def toggle_collapse(n, is_open):
    fig = go.Figure()
    if n % 2 != 0:
        fig = plot_gpt(plot_dict["query"],plot_dict["title"])
        return not is_open, fig, [html.I(className="bi bi-graph-down-arrow", style=dict(display='inline-block'))," Collapse the Plot"], False
    elif n == 0:
        return is_open, fig, [html.I(className="bi bi-graph-up-arrow", style=dict(display='inline-block'))," Generate a Plot"], False
    else:
        return not is_open, fig, [html.I(className="bi bi-graph-up-arrow", style=dict(display='inline-block'))," Regenerate the Plot"], False

# Run the app
if __name__ == '__main__':
    app.run_server(debug=False)
