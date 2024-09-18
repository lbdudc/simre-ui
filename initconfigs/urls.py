from django.urls import path
from django.contrib import admin
from django.contrib.staticfiles.urls import staticfiles_urlpatterns

admin.autodiscover()

import requirements.views


urlpatterns = [
    path("", requirements.views.index, name="index"),
    path('analisis_req/', requirements.views.analisis_req, name='analisis_req'),
]
urlpatterns += staticfiles_urlpatterns()
