from .views import result,home
from django.urls import path

urlpatterns = [
    path('', home,name='index'),
    path('result/', result, name='result'),
    
]
