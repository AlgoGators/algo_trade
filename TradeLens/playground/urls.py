from django.urls import path
from django.conf import settings
from django.conf.urls.static import static
from .views import *

# URL CONFIG
urlpatterns = [

    # menu links
    path('', dashboard, name='dashboard'),

    # Redirects
    #path('', , name=''),       ('url', view, name='html reference')
    
]
if settings.DEBUG:
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)