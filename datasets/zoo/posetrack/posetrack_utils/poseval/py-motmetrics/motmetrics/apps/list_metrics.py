
if __name__ == '__main__':
    import motmetrics
    
    mh = motmetrics.metrics.create()
    print(mh.list_metrics_markdown())