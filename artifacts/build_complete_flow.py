from pathlib import Path
import xml.etree.ElementTree as E
ROOT=Path(__file__).resolve().parent
xml=E.Element('mxfile',host='app.diagrams.net',type='device',version='24.7.17')
COLORS=[('#E7F3EF','#42967B'),('#E9F0FC','#6D8DC4'),('#EEEAFE','#9381C6'),('#FFF2DC','#C99A4E'),('#E7F4F7','#5B9FAE')]
class Page:
 def __init__(self,id,title,lanes,rows,subtitle):
  self.id=id;self.edge_count=0;self.pos={};self.cols=len(lanes);self.width=70+390*self.cols;self.height=230+rows*100
  d=E.SubElement(xml,'diagram',id=id,name=title)
  m=E.SubElement(d,'mxGraphModel',grid='1',gridSize='10',guides='1',tooltips='1',connect='1',arrows='1',fold='1',page='1',pageScale='1',pageWidth=str(self.width),pageHeight=str(self.height),background='#FFFFFF',shadow='0')
  self.r=E.SubElement(m,'root');E.SubElement(self.r,'mxCell',id='0');E.SubElement(self.r,'mxCell',id='1',parent='0')
  self.box('title',title,30,20,self.width-60,45,('#FFFFFF','#FFFFFF'),'text;',22)
  self.box('subtitle',subtitle,30,70,self.width-60,45,('#FFFFFF','#FFFFFF'),'text;',13)
  for i,(name,file) in enumerate(lanes):
   c=COLORS[i%5]
   self.box('bg'+str(i),'',35+i*390,130,370,self.height-170,('#FAFBFD','#DFE5EC'))
   self.box('head'+str(i),name+'\n'+file,45+i*390,140,350,70,c,font=15)
 def box(self,id,text,x,y,w,h,color=COLORS[0],shape='rounded=1;arcSize=10;',font=14,link=None):
  self.pos[id]=(x,y,w,h)
  attrs=dict(id=id,value=text,style=f'{shape}whiteSpace=wrap;html=0;fillColor={color[0]};strokeColor={color[1]};fontColor=#243444;fontSize={font};fontFamily=Helvetica;spacing=9;align=center;verticalAlign=middle;',vertex='1',parent='1')
  if link:attrs['link']='data:page/id,'+link
  c=E.SubElement(self.r,'mxCell',**attrs);E.SubElement(c,'mxGeometry',x=str(x),y=str(y),width=str(w),height=str(h),**{'as':'geometry'})
 def n(self,id,col,row,text,kind='',link=None):
  self.box(id,text,65+col*390,230+row*100,310,70,COLORS[col%5], 'rhombus;' if kind=='decision' else 'rounded=1;arcSize=10;',13,link)
 def e(self,a,b,label='',mode='',side=None):
  self.edge_count+=1
  ax,ay,aw,ah=self.pos[a];bx,by,bw,bh=self.pos[b]
  points=[];style=''
  if side:
   xx=side
   points=[(xx,ay+ah/2),(xx,by+bh/2)]
   aa='1' if xx>ax+aw else '0';bb='1' if xx>bx+bw else '0'
   style=f'exitX={aa};exitY=0.5;entryX={bb};entryY=0.5;'
  elif ax!=bx and ay!=by:
   yy=by-15 if by>ay else ay+ah+15
   points=[(ax+aw/2,yy),(bx+bw/2,yy)]
   style='exitX=0.5;exitY=1;entryX=0.5;entryY=0;'
  else:style=''
  color='#B25B60' if mode=='error' else ('#8C73B7' if mode=='return' else '#587383')
  c=E.SubElement(self.r,'mxCell',id='edge'+str(self.edge_count),value=label,edge='1',parent='1',source=a,target=b,style=f'edgeStyle=orthogonalEdgeStyle;rounded=1;jettySize=auto;html=0;endArrow=block;endFill=1;strokeColor={color};fontColor={color};fontSize=11;labelBackgroundColor=#FFFFFF;'+('dashed=1;' if mode in ('return','async') else '')+style)
  g=E.SubElement(c,'mxGeometry',relative='1',**{'as':'geometry'})
  if points:
   ar=E.SubElement(g,'Array',**{'as':'points'})
   for x,y in points:E.SubElement(ar,'mxPoint',x=str(x),y=str(y))
 def chain(self,*ids):
  for a,b in zip(ids,ids[1:]):self.e(a,b)

p=Page('overview','01｜完整调用总览 · 按组件阅读', [('调用入口','examples/run_bus.py'),('AgentBus','src/interaction/bus.py'),('ACP 路由层','src/agent/server.py · context.py'),('PlanningAgent','src/agent/planning_agent.py'),('执行 Agent 与依赖','tool_calling_agent.py · TCP / SCP')],17,'从上到下阅读｜实线：调用 / 控制流；虚线：异步交接 / 返回｜详细分支见后续页面，组件标题注明代码归属。')
for args in [
 ('a',0,0,'01  构造 Task 与 SessionContext\nawait bus.submit(...)'),('b',1,0,'02  submit()\n补齐会话，调用 _ensure_session()'),('c',1,1,'03  _ensure_session()\n加锁；创建或复用队列、上下文和 worker'),('d',1,2,'04  submit()\n创建 TASK 和 Future；标记 RUNNING；入队'),('w',0,2,'调用方仍在等待 bus.submit\nsubmit 内部 await future'),('e',1,3,'05  _session_worker()\nqueue.get() → _run_planner_loop()'),('f',1,4,'06  每轮 _call_planner_raw()\n原始任务 + Agent 能力 + 历史 + 上轮结果'),('g',2,4,'07  acp(name="planning", ...)\nACPServer.__call__()'),('h',2,5,'08  AgentContextManager.__call__()\nget_info(name) → 已有 agent_instance'),('i',3,5,'09  PlanningAgent.__call__()\n回填结果；组装提示词；调用模型'),('j',3,6,'10  model_manager → PlanDecision\n拆解子任务、选择 Agent、判断依赖\n记录 plan.md，返回 AgentResponse'),('k',1,7,'11  _call_planner_raw 提取 decision\n_run_planner_loop 检查完成与分派'),('l',1,8,'12  PlanDecision.dispatches\n校验名称；每项构造 PLAN 消息'),('m',1,9,'13  asyncio.gather(...)\n每个子任务调用 _call_agent(...)'),('n',2,9,'14  acp(name=子 Agent, ...)\nACPServer → AgentContextManager'),('o',4,9,'15  ToolCallingAgent.__call__()\n当前配置的主要执行 Agent'),('q',4,10,'16  模型 → actions → TCP / SCP\n实际执行，结果反馈，再决定下一步'),('r',4,11,'17  done / 达到 max_steps\n记忆与 tracer 收尾 → AgentResponse'),('s',1,12,'18  _call_agent 封装 BusMessage\ngather 等待本轮全部执行结束'),('t',1,13,'19  收集 success / result / error\n更新 round_results 和 execution_history'),('u',3,13,'20  下一轮再次调用 planning\n评价执行结果；继续、重试或结束'),('v',1,14,'21  完成或终止：_resolve_future()\n按 correlation_id 找到 Future 并赋值'),('x',0,15,'22  submit 恢复\nERROR → FAILED；否则 → DONE\nreturn BusMessage'),('y',0,16,'23  run_bus.py 读取 payload\n输出 success / result / error 和事件日志'),('z',1,16,'worker：finally queue.task_done()\n继续等待同一 session 的下一条消息')]:p.n(*args)
p.chain('a','b','c','d');p.e('d','w','await future','async');p.e('d','e','队列交接','async');p.chain('e','f','g','h','i','j');p.e('j','k','AgentResponse 经 ACP 原路返回','return');p.chain('k','l','m','n','o','q','r');p.e('r','s','AgentResponse 经 ACP 原路返回','return');p.chain('s','t','u');p.e('u','f','有剩余轮次：重新规划','return',side=1205);p.e('k','v','is_done / 终止错误',side=440);p.e('t','v','轮数耗尽','error');p.e('v','x','Future 唤醒','return');p.chain('x','y');p.e('v','z','循环退出后；超限有额外收尾')

p=Page('session','02｜提交、会话队列与结果回传',[('调用方与 submit','run_bus.py · bus.py:222'),('_ensure_session','bus.py:361'),('_session_worker 与 Future','bus.py:392 · 935')],14,'同一 session：顶层任务按队列串行；不同 session：worker 可并发。_ensure_session 返回不代表任务执行完成。')
for a in [('a',0,0,'run_bus.py：await bus.submit(...)'),('b',0,1,'缺省 session_ctx → SessionContext()\ntask.session_id 为空 → 绑定 ctx.id'),('c',1,1,'_ensure_session：async with lock\n检查 session_id 是否已有队列'),('d',1,2,'若没有：创建 asyncio.Queue\n保存 SessionContext'),('e',1,3,'asyncio.create_task(_session_worker)\n保存 worker 句柄'),('f',1,4,'已有则复用；释放锁并返回'),('g',0,5,'构造 TASK 消息\ncontent / files / recipients=[planning]\npayload.max_rounds = max_rounds'),('h',0,6,'创建 Future\n_pending_responses[correlation_id] = future'),('i',0,7,'task.mark_running()\nawait queue.put(msg)；记录 message_enqueued'),('j',0,8,'response = await future\n当前协程挂起，让出执行权'),('k',2,5,'worker while True\nmessage = await queue.get()'),('l',2,6,'await _run_planner_loop(message)\n完整规划与执行见第 03 / 04 页'),('m',2,7,'捕获未处理 Exception\n构造 ERROR 并 _resolve_future()'),('n',2,9,'_resolve_future(correlation_id, response)\npop 映射；Future 未完成才 set_result'),('o',0,10,'submit 获得 BusMessage\nERROR → mark_failed()\n否则 → mark_done()'),('q',0,11,'return response → run_bus.py\n读取 payload 并打印结果'),('r',2,11,'worker finally：queue.task_done()\n不管成功、失败或异常都确认处理完成'),('s',1,12,'submit 等待期间被取消：\nmark_cancelled() 并重新抛出\n其他 Exception：mark_failed() 并抛出'),('t',2,13,'worker 收到 CancelledError：退出循环\n取消 submit 本身不会自动停止 worker')]:p.n(*a)
p.chain('a','b','c','d','e','f','g','h','i','j');p.e('c','f','已有：复用',side=805);p.e('e','k','启动','async');p.e('i','k','入队 / 取出','async');p.chain('k','l');p.e('l','m','异常','error');p.e('m','n');p.e('l','n','循环内部交付最终消息','return',side=1190);p.e('n','o','唤醒','return');p.chain('j','o','q');p.e('l','r','执行结束后进入 finally',side=820);p.e('r','k','下一条消息',side=1210);p.e('j','s','等待中异常 / 取消','error');p.e('r','t','worker 被取消','error')

p=Page('planner','03｜规划决策、并发分派与轮次反馈',[('AgentBus：规划循环','bus.py:441 · 799 · 849'),('ACP：通用 Agent 调用','server.py:283 · context.py:1250'),('PlanningAgent：单轮规划','planning_agent.py:565')],19,'Planner 负责决策和计划记录；Bus 负责轮次控制、调度与结果收集。每轮 dispatches 中的子任务并发，依赖关系通过分轮实现。')
for a in [('a',0,0,'_run_planner_loop(message)\n先检查 TTL；过期则交付 ERROR 并返回'),('b',0,1,'读取任务、附件、ctx 和 max_rounds\n生成 agent_contract（排除 planner）\n初始化历史和 round_results'),('c',0,2,'for round_num in 1..max_rounds\n_call_planner_raw(task, files, ctx, ...)'),('d',1,2,'acp(name=self.planner_name, input, ctx)\nACPServer → AgentContextManager'),('e',1,3,'get_info(name) 获取配置与已有实例\nawait agent_instance(**input, ctx, **kwargs)'),('f',2,3,'PlanningAgent.__call__()\n获取 / 复用 PlanFile\n回填上一轮 round_results'),('g',2,4,'prompt_manager.get_messages()\n系统：Agent 能力\n任务：原始目标、轮数、累计执行历史'),('h',2,5,'model_manager(response_format=PlanDecision)\n模型拆解任务、选择 Agent\n生成本轮 dispatches 或最终答案'),('i',2,6,'更新分析与 PlanRound / 完成状态\nawait plan_file.save()\n写入 <session_id>.plan.md'),('j',2,7,'返回 AgentResponse\nextra.data = {decision, plan_path}'),('k',0,8,'_call_planner_raw 提取 decision\n调用异常 / 缺失 decision → None'),('l',0,9,'决策分流\nis_done → 最终 RESPONSE\nNone / 未完成却无分派 → ERROR'),('m',0,10,'校验 dispatches.agent_name\n精确匹配或名称解析；跳过无效项'),('n',0,11,'每个有效 dispatch 构造 PLAN 消息\ncontent = task；files = files\n单项单播，不再进入 session 队列'),('o',0,12,'asyncio.gather(return_exceptions=True)\n并发 _call_agent(name, message, ctx)'),('q',1,12,'每项：acp(name=子 Agent, ...)\n同一套 ACP / ContextManager 路由'),('r',2,12,'执行对应子 Agent.__call__()\n当前主要为 ToolCallingAgent\n详情见第 04 页'),('s',0,13,'_call_agent 封装执行结果\n正常：RESPONSE，含 success/result/error\nException：ERROR'),('t',0,14,'gather 等待全部结束\n按 dispatch 与 response 配对\n构建 round_results，追加历史摘要'),('u',0,15,'未完成且仍有轮次 → 下一轮\n失败结果也交给 Planner 决定下一步'),('v',1,16,'耗尽 max_rounds\n先 _resolve_future(ERROR) 交付错误'),('w',2,16,'额外调用一次 Planner：round=max+1\n传入最后结果，返回决策不再执行\n不会改变已交付的 ERROR'),('x',0,17,'终止分支统一 _resolve_future()\n结果通过 Future 交付，不靠本函数 return'),('y',1,18,'本函数退出 → worker finally\nqueue.task_done() → 消费下一任务')]:p.n(*a)
p.chain('a','b','c','d','e','f','g','h','i','j');p.e('j','k','经 ACP 返回','return');p.chain('k','l','m','n','o','q','r');p.e('r','s','AgentResponse','return');p.chain('s','t','u');p.e('u','c','更新历史与上轮结果','return',side=40);p.e('m','u','全部名称无效：记入历史，不执行',side=390);p.e('l','x','终止结果',side=420);p.e('a','x','TTL 过期','error',side=20);p.chain('u','v','w');p.e('w','y','收尾结束');p.e('x','y','普通终止后退出')

p=Page('execution','04｜子 Agent、工具 / 技能与执行反馈',[('Bus 与 ACP','bus.py:849 · agent/server.py'),('ToolCallingAgent','src/agent/tool_calling_agent.py'),('模型、工具、技能与记录','model_manager · TCP · SCP · memory')],16,'外层：Bus 同轮多个子任务并发。内层：ToolCallingAgent 每步的 actions 按顺序执行。done 只结束子任务，整项任务由 Planner 判断。')
for a in [('a',0,0,'_call_agent(agent_name, PLAN消息, ctx)\n记录 agent_dispatched'),('b',0,1,'acp(name=agent_name, input={task, files})\nACPServer → AgentContextManager\n找到已有实例并 await __call__'),('c',1,1,'ToolCallingAgent.__call__()\n复用 ctx，取得 tracer / record'),('d',1,2,'有附件：提取内容并增强任务描述\n启用记忆时：start_session + TASK_START\n_get_messages() 组装上下文'),('e',1,3,'while step_number < max_steps\nawait _think_and_tool(...)'),('f',2,3,'model_manager(response_format=ThinkOutput)\n返回 actions：type / name / args'),('g',1,4,'解析 actions，逐项依次执行\n解析 JSON args；按 action.type 路由'),('h',2,5,'type == skill → scp(name, input, ctx)\nSCPServer → SkillContextManager\n返回 SkillResponse'),('i',2,6,'其他 → tcp(name, input, ctx)\nTCPServer → ToolContextManager\n执行工具实例，返回 ToolResponse'),('j',1,7,'读取结果并记录 action_results\n工具名为 done：done=True，取最终结果\n跳出本步 actions 循环'),('k',2,8,'可选：memory_manager.add_event(TOOL_STEP)\n记录本步动作和输出\n异常由 _think_and_tool 捕获并记录'),('l',1,9,'step_number += 1\n保存 tracer；重新 _get_messages()\n用最新记录 / 记忆构建后续上下文'),('m',1,10,'response.done 为真 → 退出步骤循环\n否则继续，直到 max_steps'),('n',1,11,'步骤循环结束\n若 step_number >= max_steps\n当前实现覆盖为 done=False 的超限结果'),('o',2,12,'启用记忆时：TASK_END → end_session\n等待记忆收尾；最终保存 tracer'),('q',1,13,'返回 AgentResponse\nsuccess=response.done\nmessage=response.result；extra.data=response'),('r',0,13,'ACP 原路返回 → _call_agent\nAgent success=False 仍封装为 RESPONSE\n普通调用异常封装为 ERROR'),('s',0,14,'Bus gather 收集全部结果\n更新历史，下一轮反馈给 Planner'),('t',1,15,'两种结束标记\n执行 Agent：done → 子任务结束\nPlanner：is_done → 原始任务结束')]:p.n(*a)
p.chain('a','b','c','d','e','f','g');p.e('g','h','skill');p.e('g','i','tool / 默认',side=805);p.e('h','j','结果','return');p.e('i','j','结果','return');p.chain('j','k','l','m','n','o','q','r','s');p.e('m','e','未 done 且还有步数','return',side=430)

p=Page('contracts','05｜数据契约、边界与代码索引',[('数据在组件间怎样流动','请求 → 决策 → 执行结果 → Future'),('当前代码的分支语义','按实现记录；不是理想化设计')],12,'补充页用于核对数据字段和异常行为，不涉及程序启动或全局初始化。图中步骤对应本次读取的工作区代码。')
notes=[
('c1',0,0,'TASK 消息：submit → session 队列\nsession_id / task_id / correlation_id\npayload：content、files、max_rounds'),
('c2',0,1,'Planner 输入：Bus → ACP → PlanningAgent\ntask、files、ctx；round_number、max_rounds\nagent_contract、execution_history、round_results'),
('c3',0,2,'PlanDecision：PlanningAgent → Bus\nanalysis、plan_update、dispatches\nis_done、final_result（另含决策依据字段）'),
('c4',0,3,'dispatches 每项：agent_name / task / files\nBus 将每项转换为 PLAN 消息\n_call_agent 将 content 映射回 Agent.task'),
('c5',0,4,'子 Agent → AgentResponse\nsuccess / message / extra\nBus 封装为 success / result / error'),
('c6',0,5,'round_results：以 Agent 名称为键\nexecution_history：累计每轮摘要\n两者随下一轮调用回到 Planner'),
('c7',0,6,'最终 BusMessage → _resolve_future\ncorrelation_id 对应 _pending_responses\nset_result 唤醒 submit；调用方读取 payload'),
('c8',0,8,'AgentBus 源码索引\nsubmit:222；ensure:361；worker:392\nplanner_loop:441；planner_raw:799\ncall_agent:849；resolve_future:935'),
('c9',0,10,'Agent 源码索引\nACPServer.__call__:283；ContextManager:1250\nPlanningAgent.__call__:565\nToolCallingAgent.__call__:349；步骤循环:435'),
('b1',1,0,'无有效 decision / 未完成且无 dispatches\nBus 立即交付 ERROR\n所有名称无效则进入下一轮，而非立即失败'),
('b2',1,1,'Planner 模型调用失败的当前兜底\n生成 is_done=True，final_result 写错误\nBus 因此仍发 success=True 的 RESPONSE'),
('b3',1,2,'达到 max_rounds：先回传 ERROR\n再额外调用 Planner 做收尾\n收尾决策不执行，也不改写已交付结果'),
('b4',1,3,'同轮多次调用同名 Agent\nround_results[name] 会被后项覆盖\n累计历史仍按分派项逐条追加摘要'),
('b5',1,4,'ToolCallingAgent 达到 max_steps\n即使最后一步刚 done，当前条件仍会\n将 response 覆盖成超限失败'),
('b6',1,5,'done 后仍可能有记忆处理 / 模型调用\nend_session 与 tracer 保存结束后\n才返回 AgentResponse，Bus 才能收集'),
('b7',1,6,'TTL 仅在开始处理顶层消息时检查\n不是整个任务执行的超时机制\nmax_rounds 与 max_steps 是不同限制'),
('b8',1,8,'当前配置 configs/bus.py\n注册 planning、tool_calling\nPlanner 根据能力描述决定分派目标'),
('b9',1,10,'plan.md 是规划记录，不是调度入口\n执行由 Bus 消费结构化 decision 驱动\nPlanner 本身不调用 Bus，也不分派子 Agent')]
for a in notes:p.n(*a)
p.chain('c1','c2','c3','c4','c5','c6','c7')
path=ROOT/'agentbus-complete-flow.drawio'
E.indent(xml);E.ElementTree(xml).write(path,encoding='utf-8',xml_declaration=True)
# Validate every page: IDs, references, bounds and foreground overlaps.
for d in xml.findall('diagram'):
 m=d.find('mxGraphModel');cells=m.findall('./root/mxCell');ids=[c.get('id') for c in cells];assert len(ids)==len(set(ids))
 boxes=[]
 for c in cells:
  if c.get('edge')=='1':assert c.get('source') in ids and c.get('target') in ids
  if c.get('vertex')=='1':
   g=c.find('mxGeometry');x,y,w,h=[float(g.get(a)) for a in ('x','y','width','height')]
   assert x>=0 and y>=0 and x+w<=float(m.get('pageWidth')) and y+h<=float(m.get('pageHeight'))
   if not c.get('id').startswith('bg'):boxes.append((c.get('id'),x,y,w,h))
 for ix,(a,x,y,w,h) in enumerate(boxes):
  for b,xx,yy,ww,hh in boxes[ix+1:]:assert not (x<xx+ww and xx<x+w and y<yy+hh and yy<y+h),(d.get('name'),a,b)
 print(d.get('name'),len(cells),'cells: bounds, references, node overlaps OK')
print(path)

# Route connectors on a grid around all foreground nodes, preserving editability.
import heapq
from collections import defaultdict
for d in xml.findall('diagram'):
 m=d.find('mxGraphModel');cells=m.findall('./root/mxCell');byid={c.get('id'):c for c in cells};rects={}
 for c in cells:
  if c.get('vertex')=='1' and not c.get('id').startswith('bg'):
   g=c.find('mxGeometry');rects[c.get('id')]=tuple(int(g.get(a)) for a in ('x','y','width','height'))
 blocked=set();step=5
 for x,y,w,h in rects.values():
  for gx in range((x-5)//step,(x+w+5)//step+1):
   for gy in range((y-5)//step,(y+h+5)//step+1):blocked.add((gx,gy))
 usage=defaultdict(int)
 def port(rect,which):
  x,y,w,h=rect
  return {'R':((x+w+10,y+h//2),'1','0.5'),'L':((x-10,y+h//2),'0','0.5'),'B':((x+w//2,y+h+10),'0.5','1'),'T':((x+w//2,y-10),'0.5','0')}[which]
 for c in cells:
  if c.get('edge')!='1':continue
  ar=rects[c.get('source')];br=rects[c.get('target')]
  if ar[1]==br[1]:out,inc=('R','L') if ar[0]<br[0] else ('L','R')
  else:out,inc=('B','T') if ar[1]<br[1] else ('T','B')
  sp,sx,sy=port(ar,out);ep,tx,ty=port(br,inc)
  start=(sp[0]//step,sp[1]//step);end=(ep[0]//step,ep[1]//step)
  maxx=int(m.get('pageWidth'))//step-2;maxy=int(m.get('pageHeight'))//step-2
  heap=[(0,0,start)];cost={start:0};prev={};found=False
  while heap:
   _,score,cur=heapq.heappop(heap)
   if score!=cost.get(cur):continue
   if cur==end:found=True;break
   for dx,dy in ((1,0),(-1,0),(0,1),(0,-1)):
    nxt=(cur[0]+dx,cur[1]+dy)
    if nxt in blocked or not(2<=nxt[0]<=maxx and 22<=nxt[1]<=maxy):continue
    new=score+1+min(usage[nxt],3)*0.12
    if new<cost.get(nxt,1e20):
     cost[nxt]=new;prev[nxt]=cur
     heapq.heappush(heap,(new+abs(nxt[0]-end[0])+abs(nxt[1]-end[1]),new,nxt))
  assert found, (d.get('name'),c.get('source'),c.get('target'))
  path=[end]
  while path[-1]!=start:path.append(prev[path[-1]])
  path.reverse()
  for pt in path:usage[pt]+=1
  turns=[path[0]]
  for i in range(1,len(path)-1):
   if (path[i][0]-path[i-1][0],path[i][1]-path[i-1][1]) != (path[i+1][0]-path[i][0],path[i+1][1]-path[i][1]):turns.append(path[i])
  turns.append(path[-1])
  style=c.get('style').split(';');style=[s for s in style if s and not s.startswith(('exitX=','exitY=','entryX=','entryY='))]
  c.set('style',';'.join(style)+f';exitX={sx};exitY={sy};entryX={tx};entryY={ty};')
  g=c.find('mxGeometry')
  for ch in list(g):g.remove(ch)
  arr=E.SubElement(g,'Array',**{'as':'points'})
  for x,y in turns:E.SubElement(arr,'mxPoint',x=str(x*step),y=str(y*step))
 print(d.get('name'),'connectors routed around nodes')
E.indent(xml);E.ElementTree(xml).write(ROOT/'agentbus-complete-flow.drawio',encoding='utf-8',xml_declaration=True)
