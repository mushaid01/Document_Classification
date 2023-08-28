from django.shortcuts import render, redirect
from .forms import PDFFileForm
from subprocess import Popen, PIPE
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
from django.contrib.auth import get_user
from django.contrib.auth.decorators import login_required


import io
from io import StringIO
import os
import glob
import docx
import comtypes.client
import sys
import string
import PyPDF2
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from nltk.stem import WordNetLemmatizer
import warnings
warnings.filterwarnings('ignore')
import pickle
from tensorflow.keras.models import load_model
from keras.utils import pad_sequences
import numpy as np
from django.core.files.base import ContentFile


savedmodel_p='static/model/LSTM.h5'
savedvect_p="static/model/tokenizer_F.pkl"
savedModel=load_model(savedmodel_p)
tknx = pickle.load(open(savedvect_p,'rb'))
from myapp.models import PDFFile
# Create your views here.
# Parsing through the sample document and extracting the textual data
def convert2txt(file):
    alltexts = []
    with open(file, 'rb') as fh:
        rsrcmgr = PDFResourceManager()
        retstr = StringIO()
        laparams = LAParams()
        device = TextConverter(rsrcmgr, retstr, laparams=laparams)
        fp = open(file, 'rb')
        interpreter = PDFPageInterpreter(rsrcmgr, device)
        password = ""
        maxpages = 0
        caching = True
        pagenos=set()
        for page in PDFPage.get_pages(fp, pagenos, maxpages=maxpages, password=password,caching=caching, check_extractable=True):
            interpreter.process_page(page)

        text = retstr.getvalue()
        alltexts.append(text)
        fp.close()
        device.close()
        retstr.close()
    return alltexts

from cryptography.fernet import Fernet
@login_required
def index(request):
    context={}
    print("hiiiiiiiiii")
    if request.method=='POST' and request.POST.get('choice')=='1' or request.POST.get('choice')=='2' or request.POST.get('choice')=='3':
        ch=request.POST.get('choice')
        print("first choice")
        print(ch)
    if request.method=='POST' and request.POST.get('choice1')=='FREE' or request.POST.get('choice1')=='LEVEL1' or request.POST.get('choice1')=='LEVEL2':
    # if request.method=='POST' and request.POST.get('choice1')=='FREE':
        ch=request.POST.get('choice1')
        print("second choice")
        print(ch)
        return redirect("myapp:fileupload")
    return render(request,"index.html")
@login_required
def filupload(request):
        context={}
        if request.method == 'POST':
            form = PDFFileForm(request.POST, request.FILES)
            if form.is_valid():
                for file in request.FILES.getlist('file'):
                    current_user = request.user
                    pdf_file = PDFFile(user=current_user,file=file)
                    pdf_file.save()
                    #################################################
                    ###############################################

                    textdata = convert2txt(pdf_file.file.path)
                    ##################################################

                    #################################################
                    seq = tknx.texts_to_sequences(textdata)
                    padded = pad_sequences(seq, maxlen=3000)
                    pred = savedModel.predict(padded)
                    labels = ['Aggreements','Deeds','Human Resource','Taxes','Valuations']
                    pred=labels[np.argmax(pred)]
                    pdf_file.prediction = pred
                    pdf_file.save()
                    ####################################################
                    pdf_reader = PyPDF2.PdfReader(pdf_file.file.path)
                    n_pages=len(pdf_reader.pages) 
                    pdf_file.num_pages = n_pages
                    pdf_file.save()
                    #####################################################

                    page = pdf_reader.pages[0]
                    page_content = page.extract_text()
                    ################################################
                    key = Fernet.generate_key()
                    pdf_file.key = key.decode('utf-8')
                    pdf_file.save()
                    print(key)
                    f = Fernet(key)
                    with open(pdf_file.file.path, 'rb') as f_in:
                        with open(pdf_file.file.path + '.enc', 'wb') as f_out:
                            f_out.write(f.encrypt(f_in.read()))
                    # Delete the original file
                    os.remove(pdf_file.file.path)
                    # Update the file field in the PDFFile model with the encrypted file
                    pdf_file.file = pdf_file.file.name + '.enc'
                    pdf_file.save()
                    ############################################
                    context.setdefault('files', []).append({
                        'page_content': page_content,
                        'pdf_file': pdf_file,
                        'pred': pred
                    })
                    context['form']=form
                return render(request,"flup.html",context)
    
        else:
            form = PDFFileForm()
        return render(request,"flup.html",{'form':form})

from django.shortcuts import render
from myapp.models import PDFFile
from django.contrib.auth import get_user
import tempfile
from collections import Counter
import plotly.express as px
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')
nltk.download('stopwords')
def my_files(request):
    context={}
    current_user = request.user
    pdf_files = PDFFile.objects.all().filter(user=current_user)
    total_files=pdf_files.count()
    for pdf_file in pdf_files:
        key = pdf_file.key.encode('utf-8')
        f = Fernet(key)
        with open(pdf_file.file.path, 'rb') as f_in:
            decrypted_data = f.decrypt(f_in.read())    
        with tempfile.NamedTemporaryFile(delete=False) as f_out:
            f_out.write(decrypted_data)
        # Read the decrypted file using PyPDF2
        with open(f_out.name, 'rb') as f:        
            pdf_reader = PyPDF2.PdfReader(f)
                # Read the first page of the PDF
            page = pdf_reader.pages[0]
            page_content = page.extract_text()
            # Split the text into words
            stop_words = set(stopwords.words('english'))
            words = word_tokenize(page_content.lower())
            words = [word for word in words if word.isalpha() and word not in stop_words]
            word_counts = Counter(words)

            top_words = word_counts.most_common(5)
            top_words_df = pd.DataFrame(top_words, columns=['Word', 'Count'])
            fig = px.bar(top_words_df, x='Word', y='Count', title='Top 5 Words without Stopwords')
            chart=fig.to_html()
            context.setdefault('files', []).append({
                'page_content': page_content,
                'pdf_file': pdf_file,
                'prediction': pdf_file.prediction,
                'num_pages': pdf_file.num_pages,
                "chart":chart
            })
            context['total_files']=total_files
    return render(request, 'my_files.html', context)