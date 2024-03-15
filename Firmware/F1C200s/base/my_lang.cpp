#include "my_lang.h"
#include "settings.h"
#include "my_font.h"


using namespace std;
#include <list>

#define LANG_HAS_RES(n) ((n) > LANG_START && (n) < LANG_END)

int glLanguage = LANG_EN;

const char* glLangPaths[LANG_END] = {
    0,
    "/resource/res/lang_cn.txt",
    "/resource/res/lang_tw.txt",
    "/resource/res/lang_en.txt",
    0,
    0,
};

const char* glLangNames[LANG_END] = {
    "English"
};

Font glFontDefault;

typedef struct _st_lang_string{
    MY_WCHAR id[STR_MAX_LEN];
    MY_WCHAR str[STR_MAX_LEN];
} st_lang_string;

list<st_lang_string*> glLangRes;

MY_WCHAR* ParseText(MY_WCHAR *text)
{
    int i, len, j;
    len = (int)wcslen(text);
    for (i = 0; i < len -3; i ++)
    {
        if ((char)text[i] != '\\'
                && (char)text[i+1] == '\\'
                && (char)text[i] != 'n')
        {
            text[i + 1] = 0x0D;
            j = i + 2;
            for (; j < len-1; j ++)
                text[j] = text[j+1];
            text[j] = 0;
            len --;
        }
    }
    return text;
}

MY_WCHAR* MY_TR(const char *_text, MY_WCHAR* _out)
{
    if (mbstowcs(_out, _text, STR_MAX_LEN) <= 0)
    {
        *_out = 0;
        return _out;
    }
#ifdef USE_DEBUG_TRACE
    my_debug("[%s]going to uni convert:%s:", __FUNCTION__, _text);
    wcsprint(_out);
#endif /* USE_DEBUG */

    if (LANG_HAS_RES(glLanguage))
    {
        list<st_lang_string *>::iterator itr;
        itr = glLangRes.begin();
        while(itr != glLangRes.end())
        {
            st_lang_string *ls = (st_lang_string*)*itr;
            if (wcscmp(ls->id, _out) == 0)
            {
#ifdef USE_DEBUG_TRACE
                my_debug("[%s]found string.", __FUNCTION__);
                wcsprint(ls->str);
#endif /* USE_DEBUG */
                wcscpy(_out, ls->str);
                return ParseText(_out);
            }
            itr++;
        }
    }
#ifdef USE_DEBUG_TRACE
    my_debug("[%s]not found string.", __FUNCTION__);
    wcsprint(_out);
#endif /* USE_DEBUG */

    return ParseText(_out);
}

void SetLang(int _lang)
{
    //if (_lang != glLanguage)
    {
        if (_lang == LANG_CH || _lang == LANG_EN)
        {
            glFontDefault.Load("ch_s.ttf");
        }
//        else if(_lang == LANG_EN)
//        {
//            glFontDefault.Load("en.ttf");
//        }
        else if(_lang == LANG_TW)
        {
            glFontDefault.Load("ch_s.ttf");
        }
    }

    glLanguage = _lang;
    LoadLangRes();
//    TranslateEvent* evt = new TranslateEvent(_lang);
//    WindowManager::AddEvent(evt);
}

int GetLang()
{
    return glLanguage;
}

const char* GetLangName()
{
    return glLangNames[glLanguage];
}

char* GetLangPath(const char* filename, char* out)
{
    sprintf(out, "%s", filename);
    return out;
}

void ClearLangRes()
{
    list<st_lang_string*>::iterator itr;
    for (itr = glLangRes.begin(); itr != glLangRes.end(); itr++)
    {
        st_lang_string* ptr = *itr;
        delete ptr;
    }
    glLangRes.clear();
}

void LoadLangRes()
{
    int i = 1;
    char path[STR_MAX_PATH];
    FILE* fp = NULL;
    MY_WCHAR line[STR_MAX_LEN];
    st_lang_string *ls;
    char* data = NULL;
    char* pos = NULL;
    int filelen = 0;

    if (!glLangRes.empty())
        ClearLangRes();
    if (LANG_HAS_RES(glLanguage))
    {
        if (!glLangPaths[glLanguage])
            return;
        GetLangPath(glLangPaths[glLanguage], path);
        LOG_PRINT("[%s]going to load string resource %s.\n", __FUNCTION__, path);
#ifdef MY_LINUX
        fp = fopen(path, "r");
#else /* MY_LINUX */
        fp = fopen(path, "rt,ccs=UNICODE");
#endif /* MY_LINUX */
        if (!fp)
        {
            LOG_PRINT("failed to open resource file %s.\n", path);
            return;
        }
#ifdef MY_LINUX
        fseek(fp, 0, SEEK_END);
        filelen = ftell(fp);
        if (filelen < 3)
            goto exit1;
        fseek(fp, 2, SEEK_SET); //FF FE
        filelen -= 2;
        data = new char[filelen];
        fread(data, 1, filelen, fp);
        pos = data;
        while(pos - data < filelen && (pos = my_fgetws(line, STR_MAX_LEN, pos)) != 0)
#else /* MY_LINUX */
        while(fgetws(line, STR_MAX_LEN, fp))
#endif /* MY_LINUX */
        {
            wtrim(line);
            if (wcslen(line) < 1)
                continue;
            if (i > 0)
            {
                ls = new st_lang_string;
                glLangRes.push_back(ls);
                wcscpy(ls->id, line);
                //wcsprint(ls->id);
                i = -i;
            }
            else
            {
                wcscpy(ls->str, line);
                //wcsprint(ls->str);
                i = -i;
            }
        }
#ifdef MY_LINUX
exit1:
#endif /* MY_LINUX */
        fclose(fp);
#ifdef MY_LINUX
        if (data)
        {
            delete []data;
            data = NULL;
        }
#endif //MY_LINUX
        LOG_PRINT("%s", "done reading string resource.\n");
    }
}

Font* MyGetDefaultFont()
{
    return &glFontDefault;
}

MyLang::MyLang()
{

}

MyLang::~MyLang()
{

}

