from django.contrib.auth import views as auth
from django.urls import path, include
from myapp import accounts as user_view
from myapp.views import index, my_files, filupload

app_name="myapp"
urlpatterns = [
    path('',index,name="home"),
    path('accounts/login/', user_view.Login, name ='login'),
    path('logout/', auth.LogoutView.as_view(template_name ='index.html'), name ='logout'),
    path('accounts/register/', user_view.register, name ='register'),
    path('files/',my_files,name='my_files'),
    path("classification/",filupload,name="fileupload")
]