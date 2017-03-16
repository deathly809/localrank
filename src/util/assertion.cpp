

#include <util/assertion.h>

#include <cstdlib>
#include <execinfo.h>

void stackTrace() {
  void *trace[16];
  char **messages = (char **)NULL;
  int i, trace_size = 0;

  
  trace_size = backtrace(trace, 16);
  messages = backtrace_symbols(trace, trace_size);
  /* skip first stack frame (points here) */
  printf("[bt] Execution path:\n");
  for (i=1; i<trace_size; ++i)
  {
    printf("[bt] #%d %s\n", i, messages[i]);

    /* find first occurence of '(' or ' ' in message[i] and assume
     * everything before that is the file name. (Don't go beyond 0 though
     * (string terminator)*/
    size_t p = 0;
    while(messages[i][p] != '(' && messages[i][p] != ' '
            && messages[i][p] != 0)
        ++p;

    char syscom[256];
    sprintf(syscom,"addr2line %p -e %.*s", trace[i], (int)p, messages[i]);
        //last parameter is the file name of the symbol
    if(int res = system(syscom)) {
      printf("system returned non-zero value: %d\n",res); 
    }
  }

  exit(0);
}

