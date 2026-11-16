// Unit-test UI state transitions with mocked elements/Plotly, not browser QA.
const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const path = require('node:path');
const root = path.resolve(__dirname, '..');
const html = fs.readFileSync(path.join(root, 'lab_web/index.html'), 'utf8');
const fixture = JSON.parse(fs.readFileSync(path.join(root, 'analysis_output/report.json'), 'utf8'));
fixture.queries = [];
const largeFixture=JSON.parse(fs.readFileSync(path.join(root,'analysis_output/large/report.json'),'utf8'));
largeFixture.queries=[];
const fixtures={small:fixture,large:largeFixture};
const comparisonFixture=JSON.parse(fs.readFileSync(path.join(root,'analysis_output/comparison.json'),'utf8'));
const feedbackFixture={cases:[],count:0,active_count:0,pending:false,counts:Object.fromEntries(['train','validation','test'].map(s=>[s,{chat:0,skill:0,singing:0,dancing:0,focus:0}])),sources:{},last_training:null,note:'模拟纠错状态',by_model:Object.fromEntries(Object.entries(fixtures).map(([k,f])=>[k,f.feedback_report]))};
const elements = new Map();
for (const match of html.matchAll(/\bid="([^"]+)"/g)) elements.set(match[1], {
  value: '', disabled: false, hidden: false, innerHTML: '', textContent: '',
  addEventListener() {}, removeAllListeners() {}, on() {}, querySelector(){return {disabled:false};},
});
Object.entries({scope:'all',perplexity:'30',seed:'42','anchor-k':'15','detail-model':'small','report-scheme':'expanded','report-threshold':'default',expected:''}).forEach(([id,v])=>elements.get(id).value=v);
const context = vm.createContext({
  console, URLSearchParams, setTimeout, clearTimeout, AbortController, plotByModel:{},
  document:{getElementById(id){assert(elements.has(id),`missing element ${id}`);return elements.get(id);},
    documentElement:{},addEventListener(){},querySelector(){return {addEventListener(){}};},querySelectorAll(){return [];},},
  window:{addEventListener(){}},getComputedStyle(){return {getPropertyValue(){return '#66b7ff';}};},
  Plotly:{async react(id,traces){context.lastTraces=traces;context.plotByModel[id]=traces;}},
  async fetch(url,options={}) {
    let result;
    const parsed=new URL(url,'http://localhost'), pathname=parsed.pathname, model=parsed.searchParams.get('model') || 'small', active=fixtures[model];
    if(url==='/api/status') result={ready:true,busy:false,message:'分析完成'};
    else if(pathname==='/api/comparison')result=comparisonFixture;
    else if(pathname==='/api/feedback'&&!options.method)result=feedbackFixture;
    else if(pathname==='/api/state') result=active;
    else if(pathname==='/api/layout') result={points:[...active.cases.map((p,i)=>({...p,kind:'case',x:i,y:i%13,is_anchor:active.anchors.some(a=>a.id===p.id)})),...active.queries.map((p,i)=>({...p,kind:'probe',x:i,y:i}))],scope:'all',perplexity:30,seed:42,query_revision:active.queries.length,trustworthiness:.9,elapsed_ms:10};
    else if(url==='/api/messages'){
      const request=JSON.parse(options.body);
      const row={mode:'chat',matched_mode:'chat',score:.8,margin:.2,accepted:true};
      const by_model={};
      for(const [key,f]of Object.entries(fixtures)){
        const head={...row,mode:'clarify',accepted:false,score_type:'probability',raw:row,clarification:'你是想执行还是讨论？'};
        const added=request.texts.map((text,i)=>({id:`probe:${f.queries.length+i+1}`,text,mode:'chat',split:'probe',expected_mode:request.expected_mode,results:{baseline:row,expanded:row,anchors:row,classifier:head},nearest_anchors:f.anchors.slice(0,5).map(a=>({...a,similarity:.8})),nearest_cases:f.cases.slice(0,5).map(a=>({...a,similarity:.8}))}));
        f.queries.push(...added);
        const labeled=f.queries.filter(q=>q.expected_mode).length;
        f.probe_report={total:f.queries.length,labeled,note:'未标注不计算准确率',schemes:Object.fromEntries(['baseline','expanded','anchors','classifier'].map(s=>[s,{count:labeled,accuracy:labeled?1:null,macro_f1:labeled?1:null,unsafe_chat_activation:0,rejection_rate:s==='classifier'?1:0,action_recall:null}]))};
        by_model[key]={added,queries:f.queries,probe_report:f.probe_report};
      }
      result={...by_model.small,by_model};
    }else if(url==='/api/real-dataset'){
      const payload=JSON.parse(JSON.parse(options.body).dataset);
      for(const f of Object.values(fixtures))f.real_report={ready:true,dataset_id:payload.dataset_id,previously_evaluated:false,test_count:1,counts:{test:{chat:1}},reports:{classifier:f.reports.classifier},near_duplicates:[],note:'仅 UI 模拟数据'};
      result={imported:1};
    }else if(url==='/api/feedback'&&options.method==='POST'){
      const request=JSON.parse(options.body);const record={...request,id:'feedback:mock',normalized_text:request.text};
      feedbackFixture.cases=[record];feedbackFixture.count=1;feedbackFixture.pending=true;
      feedbackFixture.counts[request.split][request.mode]=1;feedbackFixture.sources[request.source]=1;
      for(const f of Object.values(fixtures))for(const q of f.queries)if(q.text===request.text)q.expected_mode=request.mode;
      result={record,created:true,feedback:feedbackFixture};
    }else if(url==='/api/feedback/train'){
      feedbackFixture.active_count=feedbackFixture.count;feedbackFixture.pending=false;
      feedbackFixture.last_training={trained_at:new Date().toISOString(),note:'模拟训练仅回归比较',models:Object.fromEntries(Object.entries(fixtures).map(([k,f])=>[k,{synthetic_before:f.reports.classifier,synthetic_after:f.reports.classifier}]))};
      for(const f of Object.values(fixtures))f.classifier.train_count+=1;
      result=feedbackFixture;
    }else throw new Error(`unexpected API ${url}`);
    return {ok:true,async json(){return result;}};
  },
});
vm.runInContext(fs.readFileSync(path.join(root,'lab_web/app.js'),'utf8'),context);
(async()=>{
  for(let i=0;i<100&&elements.get('send').disabled;i++) await new Promise(r=>setTimeout(r,10));
  // Initially HTML disabled state is represented via the mock value below only
  // after async initialization, so explicitly finish the same initialization.
  await vm.runInContext('watchBuild()',context);
  assert.equal(elements.get('send').disabled,false);
  assert(elements.get('anchors-table').innerHTML.includes('代表原句'));
  assert.equal(context.lastTraces.length,5);
  assert.equal(context.plotByModel.scatter.length,5);
  assert.equal(context.plotByModel['scatter-large'].length,5);
  assert(html.includes('<option value="focus">专注模式</option>'));
  assert(elements.get('comparison-summary').innerHTML.includes('large'));
  assert(elements.get('classifier-comparison').innerHTML.includes('按意图拒识'));
  assert(elements.get('classifier-policy').innerHTML.includes('概率下限'));
  assert(elements.get('real-note').textContent.includes('尚未通过本页导入'));
  const originalCount=fixture.cases.length;
  await assert.rejects(vm.runInContext('submitTexts([],null)',context));
  await assert.rejects(vm.runInContext('submitTexts(["有效", ""],null)',context));
  await assert.rejects(vm.runInContext('submitTexts(["有效"],"unknown")',context));
  await vm.runInContext('submitTexts(["<script>alert(1)</script>"],null)',context);
  assert(elements.get('probe-result').innerHTML.includes('&lt;script&gt;'));
  assert(!elements.get('probe-result').innerHTML.includes('<script>'));
  assert(elements.get('selection').innerHTML.includes('&lt;script&gt;'));
  assert.equal(context.lastTraces.at(-1).marker.symbol,'star');
  assert(context.lastTraces.at(-1).customdata[0][0].includes('&lt;script&gt;'));
  assert(elements.get('probe-table').innerHTML.includes('没有标注'));
  await vm.runInContext('submitTexts(["你好"],"chat")',context);
  assert.equal(fixture.probe_report.labeled,1);
  assert(elements.get('probe-table').innerHTML.includes('100.0%'));
  assert.equal(fixture.cases.length,originalCount);
  assert.equal(elements.get('send').disabled,false);
  assert(elements.get('probe-result').innerHTML.includes('同一句模型对照'));
  assert(elements.get('probe-result').innerHTML.includes('需澄清'));
  await vm.runInContext('changeModel("large")',context);
  assert(elements.get('model-meta').textContent.includes('1024'));
  assert(elements.get('download-report').href.includes('model=large'));
  assert.equal(context.plotByModel.scatter.at(-1).marker.symbol,'star');
  assert.equal(context.plotByModel['scatter-large'].at(-1).marker.symbol,'star');
  elements.get('report-scheme').value='classifier';elements.get('report-threshold').value='calibrated';
  vm.runInContext('renderReport();renderComparison()',context);
  assert(elements.get('matrix').innerHTML.includes('需澄清'));
  await vm.runInContext('importRealDataset(JSON.stringify({dataset_id:"<script>asr</script>",cases:[]}))',context);
  assert(elements.get('real-note').textContent.includes('<script>asr</script>')); // textContent is safe
  assert(elements.get('real-summary').innerHTML.includes('全量准确率'));
  assert.equal(elements.get('import-real').disabled,false);
  assert.equal(fixture.cases.length,originalCount);
  const beforeTrainCount=largeFixture.classifier.train_count;
  assert.equal(elements.get('feedback-editor').hidden,false);
  await assert.rejects(vm.runInContext('saveCorrection({text:"你好",mode:"unknown",split:"train",source:"manual",family:"hello"})',context));
  await vm.runInContext('saveCorrection({text:"你好",mode:"chat",split:"train",source:"manual",family:"hello"})',context);
  assert.equal(largeFixture.classifier.train_count,beforeTrainCount);
  assert(elements.get('feedback-note').textContent.includes('未应用'));
  assert.equal(elements.get('train-feedback').disabled,false);
  await vm.runInContext('trainCorrections()',context);
  assert.equal(largeFixture.classifier.train_count,beforeTrainCount+1);
  assert(elements.get('feedback-run').innerHTML.includes('训练前'));
  assert(elements.get('feedback-run').innerHTML.includes('训练后'));
  assert.equal(elements.get('train-feedback').disabled,true);
  assert.equal(fixture.cases.length,originalCount);
  assert.equal(elements.get('save-feedback').disabled,false);
  vm.runInContext('setupFeedbackEditor({id:"feedback:test",text:"冻结原句",mode:"chat"},{id:"feedback:test",text:"冻结原句",mode:"chat",split:"test",source:"manual",family:"frozen"})',context);
  assert.equal(elements.get('save-feedback').disabled,true);
  assert.equal(elements.get('feedback-mode').disabled,true);
  console.log('[OK] Paired charts, classifier/clarify matrix, model switching, escaping, labeling and ASR import UI transitions.');
})().catch(e=>{console.error(e);process.exitCode=1;});
